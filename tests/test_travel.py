"""Independent graph oracles and operational boundary cases for travel models."""

from copy import deepcopy
from dataclasses import replace
from itertools import permutations
import tempfile
from pathlib import Path
import unittest

import numpy as np

from xinjiang_decision.data_pipeline import Spot, number, flag, opening_intervals, unique_index, rows
from xinjiang_decision.network import TransportNetwork, TravelMatrix
from xinjiang_decision.itinerary import CalendarPolicy, schedule_route, audit_itinerary, solve_personal, solve_two_year, improve_order, route_total
from xinjiang_decision.simulation import simulate_route, wilson_interval, mean_one_lognormal


def instance(count=6):
    spots = [Spot(f"P{i}", f"Site {i}", "region", 2., 20., 1., ((9., 18.),)) for i in range(1, count + 1)]
    rng = np.random.default_rng(18)
    hours = rng.uniform(.2, 2, size=(count + 1, count + 1))
    np.fill_diagonal(hours, 0)
    return spots, TravelMatrix(("DEPOT",) + tuple(s.spot_id for s in spots), hours, hours * 10, {}, 10.)


class InputTests(unittest.TestCase):
    def test_rejects_nonfinite_negative_and_missing(self):
        for value in ("nan", "inf", "-1", "", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                number(value, "test")

    def test_boolean_is_not_python_string_truthiness(self):
        self.assertFalse(flag("False"))
        self.assertTrue(flag("true"))
        with self.assertRaises(ValueError):
            flag("maybe")

    def test_duplicate_primary_keys_rejected(self):
        with self.assertRaises(ValueError):
            unique_index([{"id": "a"}, {"id": "a"}], "id")

    def test_split_opening_intervals_preserved(self):
        window = opening_intervals({"spot_id": "P", "raw_open_time": "10:00-14:00、16:00-19:00", "open_time": "10:00", "close_time": "14:00"})
        self.assertEqual(window, ((10., 14.), (16., 19.)))

    def test_bad_clock_is_rejected(self):
        with self.assertRaises(ValueError):
            opening_intervals({"spot_id": "P", "raw_open_time": "", "open_time": "25:00", "close_time": "26:00"})

    def test_csv_bom_and_whitespace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("\ufeffid,value\na, 4 \n", encoding="utf-8")
            self.assertEqual(rows(path), [{"id": "a", "value": "4"}])


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.edges = [{"source": a, "target": b, "hours": t, "cost": c, "edge_id": str(i), "mode": "bus"}
                      for i, (a, b, t, c) in enumerate([("A", "B", 2., 5.), ("B", "C", 2., 5.), ("A", "C", 1., 30.), ("C", "A", 3., 2.)])]

    def test_dijkstra_matches_simple_path_enumeration(self):
        network = TransportNetwork(self.edges)
        lookup = {(e["source"], e["target"]): e for e in self.edges}
        for weight in (0., 5., 20.):
            best, _ = network.shortest_paths("A", weight)
            oracle = []
            for intermediates in ((), ("B",)):
                path = ("A",) + intermediates + ("C",)
                edges = [lookup[a, b] for a, b in zip(path, path[1:])]
                t, c = sum(e["hours"] for e in edges), sum(e["cost"] for e in edges)
                oracle.append((c + weight * t, t, c))
            self.assertEqual(best["C"], min(oracle))

    def test_cost_and_time_belong_to_same_path(self):
        result, _ = TransportNetwork(self.edges).shortest_paths("A", 0)
        self.assertEqual(result["C"][1:], (4., 10.))
        self.assertNotEqual(result["C"][1:], (1., 10.))

    def test_disconnected_matrix_rejected(self):
        network = TransportNetwork([{"source": "D", "target": "SPOT::X", "hours": 1., "cost": 1.}])
        with self.assertRaisesRegex(ValueError, "Unreachable"):
            network.matrix(["X"], depot="D")

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            TransportNetwork([{**self.edges[0], "cost": -1}])


class CalendarTests(unittest.TestCase):
    def test_daily_active_and_split_opening_windows(self):
        spots, matrix = instance(1)
        spots[0] = replace(spots[0], visit_hours=9., windows=((10., 14.), (16., 19.)))
        policy = CalendarPolicy(5, 1, active_hours=6)
        result = schedule_route([1], spots, matrix, policy)
        self.assertGreater(result["days"], 1)
        self.assertTrue(all(audit_itinerary(result, spots, matrix, policy).values()))
        visits = [e for e in result["events"] if e["kind"] == "visit"]
        self.assertTrue(all(e["end_hour"] <= 14 or e["start_hour"] >= 16 for e in visits))

    def test_multiday_transfer_is_not_truncated(self):
        spots, matrix = instance(1)
        matrix.hours[0, 1] = 25.
        result = schedule_route([1], spots, matrix, CalendarPolicy(10, 1))
        self.assertAlmostEqual(result["travel_hours"], 25 + matrix.hours[1, 0])
        self.assertGreaterEqual(result["days"], 3)

    def test_buffer_is_a_separate_planning_constraint(self):
        spots, matrix = instance(1)
        spots[0] = replace(spots[0], visit_hours=13.)
        result = schedule_route([1], spots, matrix, CalendarPolicy(2, 1))
        self.assertTrue(result["feasible_within_horizon"])
        self.assertFalse(result["feasible_with_buffer"])

    def test_duplicate_route_rejected(self):
        spots, matrix = instance(1)
        with self.assertRaises(ValueError):
            schedule_route([1, 1], spots, matrix)

    def test_midnight_boundary_does_not_skip_a_day(self):
        spots, matrix = instance(1)
        spots[0] = replace(spots[0], visit_hours=25., windows=((0., 24.),))
        matrix.hours[:, :] = 0
        result = schedule_route([1], spots, matrix, CalendarPolicy(5, 0, 0., 24., 24.))
        self.assertEqual(result["days"], 2)

    def test_independent_audit_catches_tampering(self):
        spots, matrix = instance(1)
        result = deepcopy(schedule_route([1], spots, matrix))
        result["events"][-1]["hours"] += 1
        checks = audit_itinerary(result, spots, matrix, CalendarPolicy())
        self.assertFalse(checks["event_chronology"])
        self.assertFalse(checks["travel_recomputed"])

    def test_empty_route_has_zero_days(self):
        spots, matrix = instance(1)
        result = schedule_route([], spots, matrix)
        self.assertEqual(result["days"], 0)
        self.assertEqual(result["transport_cost_for_two"], 0)

    def test_no_daily_window_intersection_rejected(self):
        spots, matrix = instance(1)
        spots[0] = replace(spots[0], windows=((1., 3.),))
        with self.assertRaises(ValueError):
            schedule_route([1], spots, matrix)


class RoutingTests(unittest.TestCase):
    def test_personal_selection_respects_horizon(self):
        spots, matrix = instance(6)
        result, _, _ = solve_personal(spots, [matrix], CalendarPolicy(1, 0))
        self.assertLess(len(result["route"]), len(spots))
        self.assertTrue(all(result["checks"].values()))

    def test_two_year_exact_coverage_and_balance(self):
        spots, matrix = instance(6)
        result, _ = solve_two_year(spots, [matrix], CalendarPolicy(5, 1), min_sites=2, max_day_gap=1)
        flat = [s for y in result["years"] for s in y["spot_ids"]]
        self.assertCountEqual(flat, [s.spot_id for s in spots])
        self.assertTrue(all(result["checks"].values()))
        self.assertFalse(result["proved_optimal"])

    def test_directed_reversal_never_increases_objective(self):
        _, matrix = instance(5)
        route = [1, 2, 3, 4, 5]
        result = improve_order(route, matrix.costs)
        self.assertCountEqual(route, result)
        self.assertLessEqual(route_total(result, matrix.costs), route_total(route, matrix.costs))
        oracle = min(route_total(order, matrix.costs) for order in permutations(route))
        self.assertGreaterEqual(route_total(result, matrix.costs) + 1e-8, oracle)


class SimulationTests(unittest.TestCase):
    def test_seeded_simulation_reproduces(self):
        spots, matrix = instance(2)
        args = ([1, 2], spots, matrix, CalendarPolicy(5, 1))
        self.assertEqual(simulate_route(*args, repetitions=20, seed=8), simulate_route(*args, repetitions=20, seed=8))

    def test_zero_uncertainty_equals_deterministic_calendar(self):
        spots, matrix = instance(2)
        policy = CalendarPolicy(5, 1)
        deterministic = schedule_route([1, 2], spots, matrix, policy)
        result = simulate_route([1, 2], spots, matrix, policy, 10, travel_cv=0, visit_cv=0, common_cv=0)
        self.assertEqual(result["completion_day_counts"], {str(deterministic["days"]): 10})

    def test_perfect_sample_does_not_prove_certainty(self):
        low, high = wilson_interval(20, 20)
        self.assertLess(low, 1)
        self.assertAlmostEqual(high, 1)
        low, high = wilson_interval(0, 20)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, 0)

    def test_invalid_uncertainty_rejected(self):
        with self.assertRaises(ValueError):
            mean_one_lognormal(np.random.default_rng(1), -1, 3)


if __name__ == "__main__":
    unittest.main()
