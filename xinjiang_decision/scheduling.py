"""Exact three-team scheduling using subset routing and symmetry reduction.

Unlike a heuristic, this enumerates every feasible partition under the fixed
special-access policy. The routing matrix is a scenario input, not live OD data.
"""

from itertools import combinations
import math

import numpy as np

from .portfolio import read_rows, finite_number


def truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def subset_tours(matrix):
    """Return shortest depot-return tour times and parent tables for all subsets."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("Travel matrix must be a square depot-plus-sites matrix")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Travel matrix must have finite nonnegative entries")
    n = matrix.shape[0] - 1
    if n > 16:
        raise ValueError("Subset DP supports up to 16 sites; use column generation for larger instances")
    states = 1 << n
    dp = np.full((states, n), np.inf)
    parent = np.full((states, n), -1, dtype=np.int16)
    for last in range(n):
        dp[1 << last, last] = matrix[0, last + 1]
    for mask in range(1, states):
        bits = [i for i in range(n) if mask & (1 << i)]
        for last in bits:
            previous = mask ^ (1 << last)
            if not previous:
                continue
            candidates = [i for i in bits if i != last]
            values = dp[previous, candidates] + matrix[np.asarray(candidates) + 1, last + 1]
            best = int(np.argmin(values))
            dp[mask, last] = values[best]
            parent[mask, last] = candidates[best]
    lengths = np.zeros(states)
    ends = np.full(states, -1, dtype=np.int16)
    for mask in range(1, states):
        values = dp[mask] + matrix[1:, 0]
        end = int(np.argmin(values))
        lengths[mask], ends[mask] = values[end], end
    return lengths, ends, parent


def reconstruct(mask, ends, parent):
    route = []
    last = int(ends[mask])
    while last >= 0:
        route.append(last)
        previous = int(parent[mask, last])
        mask ^= 1 << last
        last = previous
    return list(reversed(route))


def optimal_partition(travel, service, special_mask, min_sites=2):
    """Exact lexicographic (makespan, imbalance, total completion) partition."""
    n = len(service)
    if min_sites < 1 or n < 3 * min_sites or special_mask < 0 or special_mask >= (1 << n):
        raise ValueError("Invalid three-team scheduling instance")
    if not np.isfinite(service).all() or min(service) < 0:
        raise ValueError("Service times must be finite and nonnegative")
    lengths, ends, parent = subset_tours(travel)
    if len(lengths) != 1 << n:
        raise ValueError("Travel and service dimensions disagree")
    subset_service = np.zeros(1 << n)
    for mask in range(1, 1 << n):
        bit = mask & -mask
        subset_service[mask] = subset_service[mask ^ bit] + service[bit.bit_length() - 1]
    completion = lengths + subset_service
    full = (1 << n) - 1
    ordinary = full ^ special_mask
    extra = ordinary
    best_key, best_masks, checked = None, None, 0
    while True:
        first = special_mask | extra
        remaining = full ^ first
        if first.bit_count() >= min_sites and remaining.bit_count() >= 2 * min_sites:
            anchor = remaining & -remaining
            second = remaining
            while second:
                third = remaining ^ second
                # Identical general teams are interchangeable; keep one ordering.
                if second & anchor and second.bit_count() >= min_sites and third.bit_count() >= min_sites:
                    checked += 1
                    values = [float(completion[m]) for m in (first, second, third)]
                    key = (round(max(values), 9), round(max(values) - min(values), 9), round(sum(values), 9), first, second)
                    if best_key is None or key < best_key:
                        best_key, best_masks = key, (first, second, third)
                second = (second - 1) & remaining
        if extra == 0:
            break
        extra = (extra - 1) & ordinary
    if best_masks is None:
        raise ValueError("No feasible three-team partition")
    groups = []
    for i, mask in enumerate(best_masks):
        groups.append({"group": i + 1, "mask": mask, "indices": reconstruct(mask, ends, parent),
                       "travel_hours": float(lengths[mask]), "service_hours": float(subset_service[mask]), "completion_hours": float(completion[mask])})
    assert (best_masks[0] & special_mask) == special_mask
    assert all(not best_masks[a] & best_masks[b] for a, b in combinations(range(3), 2))
    assert best_masks[0] | best_masks[1] | best_masks[2] == full
    return {"status": "optimal_fixed_policy", "max_completion_hours": best_key[0], "balance_gap_hours": best_key[1],
            "checked_canonical_partitions": checked, "groups": groups,
            "secondary_objective": "minimize imbalance, then total completion; group2/group3 exchange symmetry removed"}


def load_cultural_instance(sites_path, travel_path):
    sites = read_rows(sites_path)
    ids = ["DEPOT_URUMQI"] + [row["spot_id"] for row in sites]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate cultural site IDs")
    index = {name: i for i, name in enumerate(ids)}
    matrix = np.full((len(ids), len(ids)), np.nan)
    np.fill_diagonal(matrix, 0)
    for row in read_rows(travel_path):
        if row["from_id"] in index and row["to_id"] in index:
            a, b = index[row["from_id"]], index[row["to_id"]]
            value = finite_number(row["time_hours"], "time_hours")
            if np.isfinite(matrix[a, b]) and abs(matrix[a, b] - value) > 1e-9:
                raise ValueError("Conflicting OD travel labels")
            matrix[a, b] = value
    if not np.isfinite(matrix).all():
        raise ValueError("Missing cultural travel labels")
    service = np.array([finite_number(row["fieldwork_hours"], "fieldwork_hours") + finite_number(row["safety_buffer_hours"], "safety_buffer_hours") for row in sites])
    special = sum(1 << i for i, row in enumerate(sites) if truth(row["requires_approval_final"]))
    return sites, matrix, service, special


def solve_cultural_schedule(sites_path, travel_path):
    sites, matrix, service, special = load_cultural_instance(sites_path, travel_path)
    solved = optimal_partition(matrix, service, special)
    for group in solved["groups"]:
        group["site_ids"] = [sites[i]["spot_id"] for i in group["indices"]]
        group["site_names"] = [sites[i]["spot_name"] for i in group["indices"]]
    solved.update({"candidate_sites": len(sites), "approval_sites": special.bit_count(), "scope": "All sites assigned once; three Urumqi round trips; approval sites fixed to group1; no cross-group sharing"})
    return solved
