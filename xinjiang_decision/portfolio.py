"""Binary route-portfolio optimization with a separate feasibility audit.

Implementation written for this repository from the mathematical model.
The reference CSVs contain scenario data, not live tourism inventory.
"""

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
import csv
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


DEFAULT_WEIGHTS = {
    "quality": .25, "capacity": .20, "comfort": .15, "diversity": .15,
    "low_pressure": .10, "attraction": .10, "resource": -.03, "travel": -.02,
}
FEATURE_COLUMNS = {
    "quality": "portfolio_quality_score", "comfort": "comfort_score",
    "diversity": "diversity_score", "low_pressure": "low_pressure_score",
    "attraction": "attraction_score", "resource": "resource_intensity_score",
    "travel": "avg_interspot_travel_hours_proxy",
}


class OptimizationError(RuntimeError):
    """The requested policy has no usable integer solution."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def regions(value):
    return frozenset(part.strip() for part in value.replace("；", ";").split(";") if part.strip())


def finite_number(value, field):
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid numeric value for {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value for {field}")
    return number


@dataclass(frozen=True)
class Route:
    route_id: str
    product_type: str
    regions: frozenset
    capacity: float
    quality_pass: int
    features: dict
    name: str = ""


@dataclass(frozen=True)
class Policy:
    quotas: dict
    regions: frozenset
    capacity_floor: float
    quality_floor: int

    @property
    def count(self):
        return sum(self.quotas.values())


def load_routes(path):
    result = []
    seen = set()
    required = {"route_id", "product_type", "region_set", "route_capacity_persons_12day", "buffer_days", "long_transfer_days_proxy"} | set(FEATURE_COLUMNS.values())
    for row in read_rows(path):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Missing route columns: {sorted(missing)}")
        route_id = row["route_id"].strip()
        if not route_id or route_id in seen:
            raise ValueError(f"Empty or duplicate route ID: {route_id!r}")
        seen.add(route_id)
        values = {key: finite_number(row[col], col) for key, col in FEATURE_COLUMNS.items()}
        capacity = finite_number(row["route_capacity_persons_12day"], "capacity")
        if capacity < 0:
            raise ValueError("Capacity must be nonnegative")
        passed = int(values["comfort"] >= 75 and finite_number(row["buffer_days"], "buffer_days") >= 2 and finite_number(row["long_transfer_days_proxy"], "long_transfer_days_proxy") <= 3 and values["travel"] <= 6)
        result.append(Route(route_id, row["product_type"], regions(row["region_set"]), capacity, passed, values, row.get("route_theme", route_id)))
    if not result:
        raise ValueError("The candidate table is empty")
    return result


def policy_from_baseline(routes, baseline_ids, capacity_floor=106358):
    baseline_ids = list(baseline_ids)
    if len(set(baseline_ids)) != len(baseline_ids):
        raise ValueError("Baseline contains duplicate route IDs")
    lookup = {r.route_id: r for r in routes}
    if not set(baseline_ids) <= lookup.keys():
        raise ValueError("Baseline references unknown routes")
    selected = [lookup[i] for i in baseline_ids]
    if not selected:
        raise ValueError("Baseline must contain at least one route")
    return Policy(dict(Counter(r.product_type for r in selected)), frozenset().union(*(r.regions for r in selected)), float(capacity_floor), sum(r.quality_pass for r in selected))


def score_routes(routes, weights=None):
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if set(weights) != set(DEFAULT_WEIGHTS) or not all(math.isfinite(v) for v in weights.values()):
        raise ValueError("Weights must provide every supported feature with finite values")
    features = {key: np.array([r.features[key] for r in routes], dtype=float) for key in FEATURE_COLUMNS}
    features["capacity"] = np.array([r.capacity for r in routes], dtype=float)
    for key in ("capacity", "resource", "travel"):
        vector = features[key]
        span = float(np.ptp(vector))
        features[key] = np.full(len(routes), 50.) if span == 0 else 100 * (vector - vector.min()) / span
    return sum(weights[key] * features[key] for key in weights)


def audit_selection(selected, policy):
    """Recompute all business constraints without relying on the MILP matrix."""
    counts = Counter(r.product_type for r in selected)
    covered = frozenset().union(*(r.regions for r in selected))
    checks = {"unique_ids": len({r.route_id for r in selected}) == len(selected), "selected_count": len(selected) == policy.count,
              "product_quotas": counts == Counter(policy.quotas), "region_coverage": policy.regions <= covered,
              "capacity_floor": sum(r.capacity for r in selected) + 1e-6 >= policy.capacity_floor,
              "quality_floor": sum(r.quality_pass for r in selected) >= policy.quality_floor}
    return checks


def solve_portfolio(routes, policy, weights=None, time_limit=30):
    if not routes or policy.count <= 0 or any(not isinstance(v, int) or v < 0 for v in policy.quotas.values()):
        raise ValueError("Invalid candidates or product quotas")
    if not math.isfinite(policy.capacity_floor) or policy.capacity_floor < 0 or policy.quality_floor < 0:
        raise ValueError("Invalid capacity or quality floor")
    scores = score_routes(routes, weights)
    constraints, lower, upper = [], [], []

    def add(coefficients, low, high):
        constraints.append(coefficients)
        lower.append(low)
        upper.append(high)

    add(np.ones(len(routes)), policy.count, policy.count)
    add([r.capacity for r in routes], policy.capacity_floor, np.inf)
    add([r.quality_pass for r in routes], policy.quality_floor, np.inf)
    # Include candidate types with a zero quota too: an unknown type cannot slip in.
    for kind in sorted({r.product_type for r in routes} | set(policy.quotas)):
        quota = policy.quotas.get(kind, 0)
        add([int(r.product_type == kind) for r in routes], quota, quota)
    for region in sorted(policy.regions):
        add([int(region in r.regions) for r in routes], 1, np.inf)

    result = milp(c=-scores, integrality=np.ones(len(routes)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(np.asarray(constraints, dtype=float), lower, upper),
                  options={"time_limit": float(time_limit), "mip_rel_gap": 0.0})
    if result.x is None:
        raise OptimizationError(f"No feasible integer portfolio (status={result.status}): {result.message}", status=result.status)
    if np.max(np.abs(result.x - np.round(result.x))) > 1e-5:
        raise OptimizationError("Solver returned a fractional incumbent")
    chosen = [r for r, value in zip(routes, result.x) if value > .5]
    checks = audit_selection(chosen, policy)
    if not all(checks.values()):
        raise OptimizationError(f"Independent constraint audit failed: {checks}")
    selected_ids = {r.route_id for r in chosen}
    return {"status": "optimal" if result.status == 0 else "feasible_time_limit", "proved_optimal": result.status == 0,
            "score": float(sum(s for r, s in zip(routes, scores) if r.route_id in selected_ids)),
            "capacity": sum(r.capacity for r in chosen), "quality_pass_count": sum(r.quality_pass for r in chosen),
            "selected_ids": sorted(selected_ids), "mip_gap": float(result.mip_gap), "checks": checks}


def sensitivity_analysis(routes, policy, reference_ids):
    cases = {"reference_weights": DEFAULT_WEIGHTS,
             "capacity_priority": dict(DEFAULT_WEIGHTS, capacity=.35, quality=.15, comfort=.10),
             "comfort_priority": dict(DEFAULT_WEIGHTS, comfort=.30, capacity=.10, quality=.20),
             "diversity_priority": dict(DEFAULT_WEIGHTS, diversity=.30, quality=.15, capacity=.15),
             "low_pressure_priority": dict(DEFAULT_WEIGHTS, low_pressure=.25, quality=.15, capacity=.15)}
    rows = []
    for name, weights in cases.items():
        solved = solve_portfolio(routes, policy, weights)
        rows.append({"case": name, "capacity": solved["capacity"], "quality_pass_count": solved["quality_pass_count"],
                     "score_with_case_weights": solved["score"], "overlap_with_reference": len(set(reference_ids) & set(solved["selected_ids"])),
                     "selected_count": len(solved["selected_ids"]), "status": solved["status"]})
    return rows


def demand_stress_test(routes, policy):
    rows = []
    for multiplier in (1., 1.1, 1.25, 1.5, 1.75, 2.):
        floor = math.ceil(policy.capacity_floor * multiplier)
        try:
            solved = solve_portfolio(routes, replace(policy, capacity_floor=floor))
            rows.append({"demand_multiplier": multiplier, "required_capacity": floor, "status": solved["status"], "selected_capacity": solved["capacity"], "selected_score": solved["score"]})
        except OptimizationError as exc:
            if exc.status != 2:
                raise
            rows.append({"demand_multiplier": multiplier, "required_capacity": floor, "status": "infeasible", "selected_capacity": None, "selected_score": None})
    return rows
