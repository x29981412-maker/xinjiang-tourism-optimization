"""Validate source tables and retain an explicit account of every exclusion."""

from dataclasses import asdict, dataclass
import csv
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class Spot:
    spot_id: str
    name: str
    region: str
    visit_hours: float
    ticket_for_two: float
    priority: float
    windows: tuple
    restricted: bool = False
    border_permit: bool = False
    reservation: bool = False


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Invalid CSV header: {path}")
        return [{k: v.strip() if isinstance(v, str) else v for k, v in row.items()} for row in reader]


def number(value, field, minimum=0):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field}: {value!r}") from error
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"Invalid {field}: {value!r}")
    return result


def flag(value):
    if str(value).lower() not in {"true", "false"}:
        raise ValueError(f"Expected explicit boolean, got {value!r}")
    return str(value).lower() == "true"


def clock(value):
    hour, minute = map(int, value.split(":"))
    if not (0 <= hour <= 24 and 0 <= minute < 60) or (hour == 24 and minute):
        raise ValueError(f"Invalid clock: {value}")
    return hour + minute / 60


def opening_intervals(row):
    # Preserve the split opening hours that the source's first-window columns omit.
    if "、" in row["raw_open_time"]:
        pairs = re.findall(r"(\d{1,2}:\d{2})\s*[-—~]\s*(\d{1,2}:\d{2})", row["raw_open_time"])
    else:
        pairs = [(row["open_time"], row["close_time"])]
    intervals = tuple(sorted((clock(a), clock(b)) for a, b in pairs))
    if not intervals or any(a >= b for a, b in intervals):
        raise ValueError(f"Invalid opening window for {row['spot_id']}")
    if any(b > c for (_, b), (c, _) in zip(intervals, intervals[1:])):
        raise ValueError(f"Overlapping opening windows for {row['spot_id']}")
    return intervals


def unique_index(records, key):
    indexed = {}
    for row in records:
        identity = row.get(key)
        if not identity or identity in indexed:
            raise ValueError(f"Missing or duplicate {key}: {identity!r}")
        indexed[identity] = row
    return indexed


def load_inputs(directory):
    directory = Path(directory)
    spot_rows = unique_index(rows(directory / "spot_clean.csv"), "spot_id")
    windows = unique_index(rows(directory / "spot_time_windows.csv"), "spot_id")
    nodes = unique_index(rows(directory / "multimodal_nodes.csv"), "node_id")
    edge_rows = unique_index(rows(directory / "multimodal_edges.csv"), "edge_id")
    spots, excluded, notes = [], [], []
    for identity, row in spot_rows.items():
        if identity not in windows or "SPOT::" + identity not in nodes:
            raise ValueError(f"Unmatched spot/window/network key: {identity}")
        restricted = flag(row["ordinary_tourist_restricted"]) or flag(row["requires_approval"])
        spot = Spot(identity, row["spot_name"], row["region_cluster"],
                    number(row["visit_hours_mid"], "visit_hours", .001),
                    2 * number(row["ticket_high_total_yuan_per_person"], "ticket"),
                    number(row["priority_score_for_op"], "priority"), opening_intervals(windows[identity]),
                    restricted, flag(row["requires_border_permit"]), flag(row["requires_reservation"]))
        if restricted:
            excluded.append({"spot_id": identity, "name": spot.name, "reason": "ordinary-tourist restriction or special approval"})
        else:
            spots.append(spot)
        if windows[identity]["time_window_quality"] == "simulated_default":
            notes.append({"spot_id": identity, "issue": "source window is a simulated default"})
        if row["map_confidence"] == "low":
            notes.append({"spot_id": identity, "issue": "low-confidence geographical mapping retained"})
        if windows[identity]["weekly_or_seasonal_rule"]:
            notes.append({"spot_id": identity, "issue": "weekly/seasonal exception not calendar-modeled"})
    edges = []
    for identity, row in edge_rows.items():
        if row["from_node"] not in nodes or row["to_node"] not in nodes:
            raise ValueError(f"Unknown edge endpoint: {identity}")
        if row["from_node"] == row["to_node"]:
            raise ValueError(f"Self edge: {identity}")
        edges.append({"edge_id": identity, "source": row["from_node"], "target": row["to_node"],
                      "mode": row["mode"], "hours": number(row["base_time_hours"], "edge time", .000001)
                      + number(row["schedule_wait_hours"] or 0, "waiting time"),
                      "cost": number(row["base_cost_yuan_per_two"], "edge cost"),
                      "source_id": row["source_id"]})
    if len({s.spot_id for s in spots}) != len(spots):
        raise ValueError("Duplicate eligible spot")
    audit = {"source_spots": len(spot_rows), "eligible_spots": len(spots), "network_nodes": len(nodes),
             "network_edges": len(edges), "excluded": excluded, "quality_notes": notes,
             "cleaning": ["trim whitespace and UTF-8 BOM", "reject duplicate primary keys", "check foreign keys",
                          "reject missing/nonfinite/negative numeric values", "preserve split opening windows"],
             "assumptions": ["summer opening windows; weekly closures are not date-resolved",
                             "border permits and reservations obtained in advance",
                             "source schedule_wait_hours added once to each corresponding edge",
                             "source numerical costs are scenario estimates for two people, not live fares"]}
    return spots, edges, audit


def export_spots(spots):
    return [{**asdict(s), "windows": ";".join(f"{a:g}-{b:g}" for a, b in s.windows)} for s in spots]
