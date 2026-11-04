"""Seeded scenario simulation, with uncertainty intervals rather than false precision."""

import math
from statistics import NormalDist

import numpy as np

from .itinerary import schedule_route


def wilson_interval(successes, count, confidence=.95):
    if count < 1 or not 0 <= successes <= count or not 0 < confidence < 1:
        raise ValueError("Invalid binomial confidence interval parameters")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0., center - half), min(1., center + half)


def mean_one_lognormal(rng, coefficient_of_variation, size):
    if not math.isfinite(coefficient_of_variation) or coefficient_of_variation < 0:
        raise ValueError("CV must be finite and nonnegative")
    sigma = math.sqrt(math.log1p(coefficient_of_variation ** 2))
    return rng.lognormal(-sigma * sigma / 2, sigma, size)


def simulate_route(route, spots, matrix, policy, repetitions=500, seed=20261104,
                   travel_cv=.20, visit_cv=.10, common_cv=.08):
    if repetitions < 1:
        raise ValueError("At least one simulation is required")
    rng = np.random.default_rng(seed)
    days, elapsed = [], []
    for _ in range(repetitions):
        common = float(mean_one_lognormal(rng, common_cv, 1)[0])
        travel = mean_one_lognormal(rng, travel_cv, len(route) + 1) * common
        visit = mean_one_lognormal(rng, visit_cv, len(route))
        result = schedule_route(route, spots, matrix, policy, travel, visit)
        days.append(result["days"])
        elapsed.append(result["elapsed_hours"])
    successes = sum(day <= policy.horizon_days for day in days)
    low, high = wilson_interval(successes, repetitions)
    return {"repetitions": repetitions, "seed": seed, "horizon_days": policy.horizon_days,
            "travel_cv": travel_cv, "visit_cv": visit_cv, "trip_common_cv": common_cv,
            "successes": successes, "completion_rate": successes / repetitions,
            "wilson_95_low": low, "wilson_95_high": high,
            "completion_days_p50": float(np.quantile(days, .5)), "completion_days_p90": float(np.quantile(days, .9)),
            "elapsed_hours_p90": float(np.quantile(elapsed, .9)),
            "completion_day_counts": {str(day): days.count(day) for day in sorted(set(days))},
            "interpretation": "assumption-driven scenarios, not an empirical completion probability"}


def uncertainty_sweep(route, spots, matrix, policy, repetitions=500, seed=20261104):
    return [{"case": name, **simulate_route(route, spots, matrix, policy, repetitions, seed, cv, .10, common)}
            for name, cv, common in (("mild", .10, .04), ("reference", .20, .08), ("severe", .40, .16))]
