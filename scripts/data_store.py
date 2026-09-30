"""Shared format for append-only migration snapshots.

Each snapshot records signed changes from the previous snapshot in the same
calendar year. The first snapshot is the year-to-date baseline. Row dimensions:
decision = country, institution, case type, marker, change;
application = country, institution, case type, change;
status = country, institution, status, change.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "snapshots"
DICTIONARIES = ROOT / "data" / "dictionaries.json"
METRICS = {"decisions": 4, "applications": 3, "statuses": 3}


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def snapshot_path(root: Path, snapshot: dict) -> Path:
    day = datetime.fromtimestamp(snapshot["timestamp"] / 1000, timezone.utc).date()
    return root / str(snapshot["year"]) / f"{day.isoformat()}-{snapshot['id']}.json"


def iter_snapshots(root: Path = SOURCE, year: int | None = None):
    paths = (root / str(year)).glob("*.json") if year is not None else root.glob("*/*.json")
    for path in sorted(paths):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if snapshot.get("schemaVersion") != 1:
            raise ValueError(f"Unsupported snapshot schema: {path}")
        yield snapshot


def replay(snapshots):
    states = {metric: defaultdict(int) for metric in METRICS}
    for snapshot in snapshots:
        for metric, size in METRICS.items():
            for row in snapshot["changes"][metric]:
                if len(row) != size + 1:
                    raise ValueError(f"Wrong {metric} row length in snapshot {snapshot['id']}")
                *key, change = row
                key = tuple(key)
                states[metric][key] += change
                if states[metric][key] < 0:
                    raise ValueError(f"Negative count for {metric} {key} in snapshot {snapshot['id']}")
                if states[metric][key] == 0:
                    del states[metric][key]
        if sum(states["decisions"].values()) != snapshot["totals"]["decisions"]:
            raise ValueError(f"Decision total mismatch in snapshot {snapshot['id']}")
    return states


def make_changes(previous: dict, current: dict):
    return [list(key) + [current.get(key, 0) - previous.get(key, 0)]
            for key in sorted(previous.keys() | current.keys())
            if current.get(key, 0) != previous.get(key, 0)]
