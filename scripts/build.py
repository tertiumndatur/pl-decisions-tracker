#!/usr/bin/env python3
"""Build the offline static dashboard from append-only snapshot files."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from data_store import ROOT, SOURCE, DICTIONARIES, METRICS, dump_json, iter_snapshots, make_changes, replay


def repair_transient_outages(snapshots: list[dict]) -> tuple[list[dict], list[dict]]:
    """Ignore a metric only when a catastrophic drop immediately reverses.

    The raw snapshots remain untouched. Other metrics from the same observation
    are retained, and the next good observation gets the net change since the
    last published state rather than the artificial rebound.
    """
    outages = {}
    for index in range(1, len(snapshots) - 1):
        previous, current, following = snapshots[index - 1:index + 2]
        bad = [metric for metric in METRICS
               if previous["totals"][metric] >= 1000
               and current["totals"][metric] < previous["totals"][metric] * 0.5
               and following["totals"][metric] >= previous["totals"][metric] * 0.8]
        if bad:
            outages[index] = bad

    raw = {metric: {} for metric in METRICS}
    published = {metric: {} for metric in METRICS}
    repaired = []
    report = []
    for index, snapshot in enumerate(snapshots):
        changes = {}
        totals = {}
        for metric in METRICS:
            for *key, delta in snapshot["changes"][metric]:
                key = tuple(key)
                value = raw[metric].get(key, 0) + delta
                if value < 0:
                    raise ValueError(f"Negative {metric} row in snapshot {snapshot['id']}")
                if value:
                    raw[metric][key] = value
                else:
                    raw[metric].pop(key, None)
            if sum(raw[metric].values()) != snapshot["totals"][metric]:
                raise ValueError(f"{metric} total mismatch in snapshot {snapshot['id']}")
            if metric not in outages.get(index, ()):
                changes[metric] = make_changes(published[metric], raw[metric])
                published[metric] = raw[metric].copy()
            else:
                changes[metric] = []
            totals[metric] = sum(published[metric].values())

        if index in outages:
            report.append({"id": snapshot["id"], "date": datetime.fromtimestamp(
                snapshot["timestamp"] / 1000, timezone.utc).date().isoformat(),
                "metrics": outages[index]})
        if any(changes.values()) or index not in outages:
            repaired.append({**snapshot, "changes": changes, "totals": totals})
    return repaired, report


def export_year(year: int, snapshots: list[dict], target: Path,
                decision_institutions: set[int], used: dict[str, set[int]],
                used_by_metric: dict[str, set[int]]):
    # Collapse multiple imports on one UTC date into the last published point.
    by_day = {}
    for snapshot in snapshots:
        day = datetime.fromtimestamp(snapshot["timestamp"] / 1000, timezone.utc).date().isoformat()
        by_day[day] = snapshot
    days = sorted(by_day)
    index_for_day = {day: index for index, day in enumerate(days)}
    points = [{"date": day, "timestamp": by_day[day]["timestamp"], "totals": {}}
              for day in days]

    countries = defaultdict(lambda: {metric: [] for metric in METRICS})
    aggregate = {metric: defaultdict(int) for metric in METRICS}
    daily_changes = {metric: [0] * len(days) for metric in METRICS}
    for snapshot in snapshots:
        day = datetime.fromtimestamp(snapshot["timestamp"] / 1000, timezone.utc).date().isoformat()
        index = index_for_day[day]
        for metric in METRICS:
            for country, *rest in snapshot["changes"][metric]:
                *dimensions, change = rest
                institution = dimensions[0]
                if metric != "applications" and institution not in decision_institutions:
                    continue
                if metric in ("decisions", "applications") and dimensions[1] == 5:
                    continue
                countries[country][metric].append([index, *dimensions, change])
                aggregate[metric][(index, *dimensions)] += change
                daily_changes[metric][index] += change
                used["countries"].add(country)
                used["institutions"].add(institution)
                used_by_metric[metric].add(institution)
                if metric == "statuses":
                    used["statuses"].add(dimensions[1])
                else:
                    used["caseTypes"].add(dimensions[1])
                    if metric == "decisions":
                        used["decisionMarkers"].add(dimensions[2])

    for metric, changes in daily_changes.items():
        total = 0
        for index, change in enumerate(changes):
            total += change
            points[index]["totals"][metric] = total

    # A separate pre-aggregated shard makes "all citizenships" just as fast as one.
    all_rows = {metric: [[*key, change] for key, change in sorted(rows.items()) if change]
                for metric, rows in aggregate.items()}
    dump_json(target / str(year) / "all.json", all_rows)
    for country, pack in sorted(countries.items()):
        dump_json(target / str(year) / f"{country}.json", pack)
    return {"snapshots": points, "countries": sorted(countries),
            "lastUpdated": points[-1]["date"]}


def build(source: Path = SOURCE, dictionaries: Path = DICTIONARIES, destination: Path = ROOT / "dist"):
    labels = json.loads(dictionaries.read_text(encoding="utf-8"))
    missing_codes = [item["id"] for item in labels["institutions"] if not item.get("authorityCode")]
    if missing_codes:
        raise ValueError(f"Missing authorityCode for institutions: {missing_codes}")
    decision_institutions = {item["id"] for item in labels["institutions"]
                             if (item["authorityCode"] == "WOJ" and item["name"].startswith("Wojewoda "))
                             or item["id"] == 810}
    if 810 not in decision_institutions or len(decision_institutions) != 17:
        raise ValueError("Missing Szef Urzędu or voivodeship institutions in the source dictionary")
    used = {name: set() for name in labels}
    used_by_metric = {metric: set() for metric in METRICS}
    by_year = defaultdict(list)
    for snapshot in iter_snapshots(source):
        by_year[snapshot["year"]].append(snapshot)
    if not by_year:
        raise ValueError("No source snapshots found")

    with tempfile.TemporaryDirectory(prefix=".dashboard-build-", dir=ROOT) as temporary:
        staging = Path(temporary)
        metadata = {}
        for year, snapshots in sorted(by_year.items()):
            snapshots.sort(key=lambda s: (s["timestamp"], str(s["id"])))
            replay(snapshots)  # Refuse to publish a corrupt migration.
            published, outages = repair_transient_outages(snapshots)
            replay(published)
            metadata[str(year)] = export_year(year, published, staging / "data", decision_institutions,
                                              used, used_by_metric)
            if outages:
                metadata[str(year)]["omittedSourceMetrics"] = outages
                print(f"Filtered transient source outages in {year}: {outages}")
        published_labels = {name: [item for item in items if item["id"] in used[name]]
                            for name, items in labels.items()}
        manifest = {"schemaVersion": 1, "latestYear": max(by_year), "years": metadata,
                    "dictionaries": published_labels,
                    "institutionsByMetric": {metric: sorted(ids) for metric, ids in used_by_metric.items()},
                    "sourceUrl": "https://migracje.gov.pl/",
                    "note": "Snapshot changes; first snapshot of each year is a baseline. Institution groups use UdSC authorityCode. Decisions and statuses include 16 voivodes and Szef Urzędu; applications include all source institutions. Case type 5 is excluded."}
        dump_json(staging / "data" / "manifest.json", manifest)
        shutil.copy2(ROOT / "index.html", staging / "index.html")
        shutil.copytree(ROOT / "web", staging / "web")
        (staging / ".nojekyll").touch()
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(staging), str(destination))

    size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    print(f"Built {destination}: {len(by_year)} years, {size / 1_000_000:.1f} MB")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--dictionaries", type=Path, default=DICTIONARIES)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    build(args.source, args.dictionaries, args.output)
