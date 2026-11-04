"""Small exhaustive oracles exercise the actual mathematical constraints."""

from itertools import combinations, permutations, product
import unittest

import numpy as np

from xinjiang_decision.portfolio import Route, Policy, OptimizationError, audit_selection, score_routes, solve_portfolio, DEFAULT_WEIGHTS
from xinjiang_decision.scheduling import subset_tours, optimal_partition


def route(name, kind, cap, passed, area, score):
    features = {key: 50. for key in ("quality", "comfort", "diversity", "low_pressure", "attraction", "resource", "travel")}
    features["quality"] = score
    return Route(name, kind, frozenset([area]), cap, passed, features)


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.routes = [route("a1", "A", 3, 1, "north", 9), route("a2", "A", 6, 1, "north", 8),
                       route("b1", "B", 3, 0, "south", 10), route("b2", "B", 5, 1, "south", 7)]
        self.policy = Policy({"A": 1, "B": 1}, frozenset(["north", "south"]), 10, 1)
        self.weights = dict.fromkeys(DEFAULT_WEIGHTS, 0.)
        self.weights["quality"] = 1.

    def test_milp_matches_exhaustive_oracle(self):
        scores = score_routes(self.routes, self.weights)
        valid = [(sum(scores[i] for i in indices), indices) for indices in combinations(range(4), 2)
                 if all(audit_selection([self.routes[i] for i in indices], self.policy).values())]
        expected = max(valid)[0]
        solved = solve_portfolio(self.routes, self.policy, self.weights)
        self.assertAlmostEqual(solved["score"], expected)
        self.assertEqual(set(solved["selected_ids"]), {"a2", "b2"})
        self.assertTrue(solved["proved_optimal"])

    def test_infeasible_demand_is_not_a_success(self):
        with self.assertRaises(OptimizationError):
            solve_portfolio(self.routes, Policy(self.policy.quotas, self.policy.regions, 1000, 1))

    def test_audit_rejects_duplicates_and_bad_quotas(self):
        checks = audit_selection([self.routes[0], self.routes[0]], self.policy)
        self.assertFalse(checks["unique_ids"])
        self.assertFalse(checks["product_quotas"])
        self.assertFalse(checks["region_coverage"])

    def test_zero_quota_prevents_new_product_types(self):
        extra = route("c1", "C", 100, 1, "south", 1000)
        solved = solve_portfolio(self.routes + [extra], self.policy, self.weights)
        self.assertNotIn("c1", solved["selected_ids"])

    def test_missing_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            score_routes(self.routes, {"quality": 1})


class SchedulingTests(unittest.TestCase):
    def test_subset_dp_matches_all_permutations(self):
        rng = np.random.default_rng(42)
        matrix = rng.integers(1, 20, size=(6, 6)).astype(float)
        np.fill_diagonal(matrix, 0)
        lengths, _, _ = subset_tours(matrix)
        for mask in range(1, 1 << 5):
            nodes = [i + 1 for i in range(5) if mask & (1 << i)]
            expected = min(sum(matrix[a, b] for a, b in zip((0,) + order, order + (0,))) for order in permutations(nodes))
            self.assertAlmostEqual(lengths[mask], expected)

    def test_partition_matches_assignment_enumeration(self):
        rng = np.random.default_rng(7)
        matrix = rng.integers(1, 15, size=(7, 7)).astype(float)
        np.fill_diagonal(matrix, 0)
        service = np.array([5., 3., 7., 2., 4., 1.])
        lengths, _, _ = subset_tours(matrix)
        best = float("inf")
        for choices in product(range(3), repeat=4):
            assignment = (0, 0) + choices
            if any(assignment.count(group) < 2 for group in range(3)):
                continue
            times = []
            for group in range(3):
                nodes = [i for i, g in enumerate(assignment) if g == group]
                mask = sum(1 << i for i in nodes)
                times.append(lengths[mask] + sum(service[i] for i in nodes))
            best = min(best, max(times))
        solved = optimal_partition(matrix, service, 3)
        self.assertAlmostEqual(solved["max_completion_hours"], best)
        all_sites = [i for g in solved["groups"] for i in g["indices"]]
        self.assertEqual(sorted(all_sites), list(range(6)))
        self.assertTrue({0, 1} <= set(solved["groups"][0]["indices"]))

    def test_missing_travel_is_rejected(self):
        with self.assertRaises(ValueError):
            subset_tours([[0, float("nan")], [1, 0]])

    def test_too_few_sites_is_rejected(self):
        with self.assertRaises(ValueError):
            optimal_partition(np.ones((5, 5)), np.ones(4), 1)


if __name__ == "__main__":
    unittest.main()
