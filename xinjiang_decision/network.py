"""Directed multimodal shortest paths; time and fare always use the SAME path."""

from dataclasses import dataclass
import heapq
import math

import numpy as np


@dataclass(frozen=True)
class TravelMatrix:
    ids: tuple
    hours: np.ndarray
    costs: np.ndarray
    paths: dict
    value_of_time: float


class TransportNetwork:
    def __init__(self, edges):
        self.adjacency = {}
        for edge in edges:
            if any(not math.isfinite(edge[k]) or edge[k] < 0 for k in ("hours", "cost")):
                raise ValueError("Shortest paths require finite nonnegative edge weights")
            self.adjacency.setdefault(edge["source"], []).append(edge)
            self.adjacency.setdefault(edge["target"], [])

    def shortest_paths(self, source, value_of_time=0):
        if value_of_time < 0 or not math.isfinite(value_of_time):
            raise ValueError("value_of_time must be finite and nonnegative")
        if source not in self.adjacency:
            raise ValueError(f"Unknown origin: {source}")
        best, parent = {source: (0., 0., 0.)}, {}
        queue = [(0., 0., 0., source)]
        while queue:
            generalized, hours, cost, node = heapq.heappop(queue)
            if (generalized, hours, cost) != best[node]:
                continue
            for edge in self.adjacency[node]:
                target = edge["target"]
                t, c = hours + edge["hours"], cost + edge["cost"]
                candidate = (c + value_of_time * t, t, c)
                if candidate < best.get(target, (math.inf, math.inf, math.inf)):
                    best[target], parent[target] = candidate, (node, edge)
                    heapq.heappush(queue, (*candidate, target))
        return best, parent

    def matrix(self, spot_ids, value_of_time=30, depot="HUB::乌鲁木齐市"):
        ids = ("DEPOT",) + tuple(spot_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate matrix node IDs")
        nodes = (depot,) + tuple("SPOT::" + s for s in spot_ids)
        hours, costs, paths = np.zeros((len(ids), len(ids))), np.zeros((len(ids), len(ids))), {}
        for i, source in enumerate(nodes):
            best, parent = self.shortest_paths(source, value_of_time)
            for j, target in enumerate(nodes):
                if target not in best:
                    raise ValueError(f"Unreachable destination: {source} -> {target}")
                hours[i, j], costs[i, j] = best[target][1:]
                node, route = target, []
                while node != source:
                    node, edge = parent[node]
                    route.append(edge)
                paths[(i, j)] = list(reversed(route))
        return TravelMatrix(ids, hours, costs, paths, value_of_time)


def leg_rows(route, matrix, case):
    records = []
    for order, (a, b) in enumerate(zip([0] + list(route), list(route) + [0]), 1):
        path = matrix.paths[a, b]
        records.append({"case": case, "leg": order, "from_id": matrix.ids[a], "to_id": matrix.ids[b],
                        "hours": float(matrix.hours[a, b]), "cost_for_two": float(matrix.costs[a, b]),
                        "modes": ">".join(edge["mode"] for edge in path),
                        "edge_ids": ">".join(edge["edge_id"] for edge in path)})
    return records
