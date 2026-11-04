"""Calendar packing, feasible insertion search, and two-year routing heuristics.

These are feasible-solution searches, not certificates of global optimality.
Travel and visits may span days; overnight stops are abstract, not hotel bookings.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CalendarPolicy:
    horizon_days: int = 30
    buffer_days: int = 2
    day_start: float = 8.
    day_end: float = 20.
    active_hours: float = 10.

    def __post_init__(self):
        if (self.horizon_days <= self.buffer_days or self.buffer_days < 0
                or not 0 <= self.day_start < self.day_end <= 24
                or not 0 < self.active_hours <= self.day_end - self.day_start):
            raise ValueError("Invalid calendar policy")


def route_total(route, matrix):
    if not route:
        return 0.
    return float(sum(matrix[a, b] for a, b in zip([0] + list(route), list(route) + [0])))


def schedule_route(route, spots, matrix, policy=CalendarPolicy(), travel_factors=None, visit_factors=None):
    route = list(route)
    if len(route) != len(set(route)) or any(i < 1 or i > len(spots) for i in route):
        raise ValueError("A route must contain distinct valid spot indices")
    travel_factors = np.ones(len(route) + 1) if travel_factors is None else np.asarray(travel_factors)
    visit_factors = np.ones(len(route)) if visit_factors is None else np.asarray(visit_factors)
    if (travel_factors.shape != (len(route) + 1,) or visit_factors.shape != (len(route),)
            or not np.isfinite(travel_factors).all() or not np.isfinite(visit_factors).all()
            or (travel_factors <= 0).any() or (visit_factors <= 0).any()):
        raise ValueError("Delay factors must be positive, finite and match route length")
    now, active, events = policy.day_start, 0., []

    def advance_day():
        nonlocal now, active
        now = (math.floor((now - 1e-8) / 24) + 1) * 24 + policy.day_start
        active = 0.

    def pack(duration, kind, identity, leg, windows):
        nonlocal now, active
        remaining = float(duration)
        if not any(min(b, policy.day_end) > max(a, policy.day_start) for a, b in windows):
            raise ValueError(f"No opening window intersects daily operating hours: {identity}")
        while remaining > 1e-8:
            if now > 3650 * 24:
                raise ValueError("Unbounded calendar packing")
            day, hour = math.floor((now + 1e-8) / 24), now % 24
            if active >= policy.active_hours - 1e-8 or hour >= policy.day_end - 1e-8:
                advance_day()
                continue
            candidates = [(max(hour, a, policy.day_start), min(b, policy.day_end)) for a, b in windows]
            candidates = [(a, b) for a, b in candidates if b > a + 1e-8]
            if not candidates:
                advance_day()
                continue
            start, end = candidates[0]
            used = min(remaining, end - start, policy.active_hours - active)
            now = day * 24 + start + used
            events.append({"day": day + 1, "kind": kind, "spot_id": identity, "leg": leg,
                           "start_hour": start, "end_hour": start + used, "hours": used})
            active += used
            remaining -= used

    for leg, (a, b) in enumerate(zip([0] + route, route + [0])):
        pack(matrix.hours[a, b] * travel_factors[leg], "travel", matrix.ids[b], leg + 1, ((0., 24.),))
        if b:
            spot = spots[b - 1]
            pack(spot.visit_hours * visit_factors[leg], "visit", spot.spot_id, leg + 1, spot.windows)
    days = max((e["day"] for e in events), default=0)
    return {"route": route, "spot_ids": [spots[i - 1].spot_id for i in route],
            "spot_names": [spots[i - 1].name for i in route], "days": days,
            "finish_absolute_hour": now if events else 0., "elapsed_hours": now - policy.day_start if events else 0.,
            "travel_hours": sum(e["hours"] for e in events if e["kind"] == "travel"),
            "visit_hours": sum(e["hours"] for e in events if e["kind"] == "visit"),
            "transport_cost_for_two": route_total(route, matrix.costs),
            "ticket_cost_for_two": sum(spots[i - 1].ticket_for_two for i in route),
            "priority_sum": sum(spots[i - 1].priority for i in route), "events": events,
            "feasible_with_buffer": days <= policy.horizon_days - policy.buffer_days,
            "feasible_within_horizon": days <= policy.horizon_days}


def audit_itinerary(solution, spots, matrix, policy, expected_ids=None):
    """Recompute from exported events, rather than trusting feasibility flags."""
    route, events = solution["route"], solution["events"]
    visit, active = {}, {}
    window_ok, chronology_ok, previous = True, True, -math.inf
    by_id = {s.spot_id: s for s in spots}
    for event in events:
        a, b, day = event["start_hour"], event["end_hour"], event["day"]
        chronology_ok &= ((day - 1) * 24 + a >= previous - 1e-7 and b > a
                          and abs((b - a) - event["hours"]) < 1e-7)
        previous = (day - 1) * 24 + b
        window_ok &= policy.day_start - 1e-7 <= a < b <= policy.day_end + 1e-7
        active[day] = active.get(day, 0.) + event["hours"]
        if event["kind"] == "visit":
            identity = event["spot_id"]
            visit[identity] = visit.get(identity, 0.) + event["hours"]
            window_ok &= any(a >= left - 1e-7 and b <= right + 1e-7 for left, right in by_id[identity].windows)
    ids = [spots[i - 1].spot_id for i in route]
    last_day = max(active, default=0)
    return {"unique_visits": len(route) == len(set(route)),
            "eligible_only": all(not spots[i - 1].restricted for i in route),
            "coverage": expected_ids is None or set(ids) == set(expected_ids),
            "service_completed": set(visit) == set(ids) and all(abs(visit[s] - by_id[s].visit_hours) < 1e-6 for s in ids),
            "travel_recomputed": abs(sum(e["hours"] for e in events if e["kind"] == "travel") - route_total(route, matrix.hours)) < 1e-6,
            "cost_recomputed": abs(solution["transport_cost_for_two"] - route_total(route, matrix.costs)) < 1e-6,
            "opening_and_daily_windows": bool(window_ok), "event_chronology": bool(chronology_ok),
            "daily_active_limit": all(t <= policy.active_hours + 1e-6 for t in active.values()),
            "horizon_with_buffer": last_day <= policy.horizon_days - policy.buffer_days,
            "reported_days": last_day == solution["days"]}


def improve_order(route, metric, evaluate=None, max_passes=12):
    """Directed 2-opt: evaluate the FULL reversed route (asymmetric edges)."""
    route = list(route)
    for _ in range(max_passes):
        best, best_value = route, route_total(route, metric)
        for a in range(len(route) - 1):
            for b in range(a + 2, len(route) + 1):
                candidate = route[:a] + route[a:b][::-1] + route[b:]
                value = route_total(candidate, metric)
                if value < best_value - 1e-7 and (evaluate is None or evaluate(candidate)):
                    best, best_value = candidate, value
        if best == route:
            break
        route = best
    return route


def insertion_order(nodes, metric, seed=None):
    remaining = set(nodes)
    route = [] if seed is None else [seed]
    remaining.difference_update(route)
    while remaining:
        choices = []
        for node in sorted(remaining):
            for position in range(len(route) + 1):
                a = route[position - 1] if position else 0
                b = route[position] if position < len(route) else 0
                choices.append((metric[a, node] + metric[node, b] - metric[a, b], node, position))
        _, node, position = min(choices)
        route.insert(position, node)
        remaining.remove(node)
    return improve_order(route, metric)


def solve_personal(spots, matrices, policy=CalendarPolicy()):
    candidates = []
    for matrix in matrices:
        for preference in (0., .12):
            route, remaining = [], set(range(1, len(spots) + 1))
            while remaining:
                current = schedule_route(route, spots, matrix, policy)
                choices = []
                for node in sorted(remaining):
                    for position in range(len(route) + 1):
                        trial = route[:position] + [node] + route[position:]
                        scheduled = schedule_route(trial, spots, matrix, policy)
                        if scheduled["feasible_with_buffer"]:
                            incremental = scheduled["elapsed_hours"] - current["elapsed_hours"]
                            density = incremental / (1 + preference * spots[node - 1].priority)
                            choices.append((density, scheduled["transport_cost_for_two"], node, position))
                if not choices:
                    break
                _, _, node, position = min(choices)
                route.insert(position, node)
                remaining.remove(node)
            route = improve_order(route, matrix.hours, lambda r: schedule_route(r, spots, matrix, policy)["feasible_with_buffer"])
            result = schedule_route(route, spots, matrix, policy)
            result.update({"value_of_time": matrix.value_of_time, "method": "feasible insertion + directed 2-opt",
                           "proved_optimal": False, "checks": audit_itinerary(result, spots, matrix, policy)})
            candidates.append((result, matrix))
    result, matrix = min(candidates, key=lambda item: (-len(item[0]["route"]), -item[0]["priority_sum"],
                                                     item[0]["transport_cost_for_two"], item[0]["elapsed_hours"]))
    if not all(result["checks"].values()) or not result["route"]:
        raise ValueError("No audited nonempty personal itinerary")
    return result, matrix, [{"value_of_time": r["value_of_time"], "spots": len(r["route"]), "priority": r["priority_sum"],
                            "days": r["days"], "transport_cost_for_two": r["transport_cost_for_two"]} for r, _ in candidates]


def solve_two_year(spots, matrices, policy=CalendarPolicy(), min_sites=8, max_day_gap=None):
    if min_sites < 1 or 2 * min_sites > len(spots):
        raise ValueError("Invalid minimum annual coverage")
    if max_day_gap is not None and max_day_gap < 0:
        raise ValueError("Annual day gap must be nonnegative")

    def feasible_pair(solutions):
        return (all(s["feasible_with_buffer"] for s in solutions)
                and (max_day_gap is None or abs(solutions[0]["days"] - solutions[1]["days"]) <= max_day_gap))

    candidates, all_nodes = [], list(range(1, len(spots) + 1))
    for matrix in matrices:
        for seed in (None, 1, len(spots)):
            order = insertion_order(all_nodes, matrix.costs, seed)
            for cut in range(min_sites, len(spots) - min_sites + 1):
                groups = [order[:cut], order[cut:]]
                solutions = [schedule_route(r, spots, matrix, policy) for r in groups]
                if feasible_pair(solutions):
                    key = (sum(s["transport_cost_for_two"] for s in solutions),
                           max(s["days"] for s in solutions), abs(solutions[0]["days"] - solutions[1]["days"]))
                    candidates.append((key, groups, matrix))
    if not candidates:
        raise ValueError("No feasible two-year split found; heuristic failure is not a proof of infeasibility")
    _, groups, matrix = min(candidates, key=lambda item: item[0])
    baseline_cost = sum(route_total(r, matrix.costs) for r in groups)
    # Relocate a spot across years only when both annual calendars remain feasible.
    for _ in range(8):
        best, best_cost = groups, sum(route_total(r, matrix.costs) for r in groups)
        for source in (0, 1):
            target = 1 - source
            if len(groups[source]) <= min_sites:
                continue
            for index, node in enumerate(groups[source]):
                removed = groups[source][:index] + groups[source][index + 1:]
                for position in range(len(groups[target]) + 1):
                    inserted = groups[target][:position] + [node] + groups[target][position:]
                    trial = [None, None]
                    trial[source], trial[target] = removed, inserted
                    cost = sum(route_total(r, matrix.costs) for r in trial)
                    if cost < best_cost - 1e-7 and feasible_pair([schedule_route(r, spots, matrix, policy) for r in trial]):
                        best, best_cost = trial, cost
        if best is groups:
            break
        groups = best
    solutions = []
    for year, route in enumerate(groups, 1):
        other_days = schedule_route(groups[2 - year], spots, matrix, policy)["days"]
        def annual_feasible(r):
            s = schedule_route(r, spots, matrix, policy)
            return s["feasible_with_buffer"] and (max_day_gap is None or abs(s["days"] - other_days) <= max_day_gap)
        route = improve_order(route, matrix.costs, annual_feasible)
        groups[year - 1] = route
        result = schedule_route(route, spots, matrix, policy)
        result.update({"year": year, "checks": audit_itinerary(result, spots, matrix, policy)})
        solutions.append(result)
    flat = [identity for s in solutions for identity in s["spot_ids"]]
    checks = {"all_eligible_exactly_once": len(flat) == len(set(flat)) == len(spots) and set(flat) == {s.spot_id for s in spots},
              "annual_minimum": all(len(s["route"]) >= min_sites for s in solutions),
              "annual_balance": max_day_gap is None or abs(solutions[0]["days"] - solutions[1]["days"]) <= max_day_gap,
              "annual_calendars": all(all(s["checks"].values()) for s in solutions)}
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"years": solutions, "transport_cost_for_two": sum(s["transport_cost_for_two"] for s in solutions),
            "initial_split_cost_for_two": baseline_cost, "checks": checks, "value_of_time": matrix.value_of_time,
            "feasible_splits_examined": len(candidates), "proved_optimal": False,
            "max_annual_day_gap": max_day_gap,
            "method": "multistart insertion + split enumeration + cross-year relocation + directed 2-opt"}, matrix
