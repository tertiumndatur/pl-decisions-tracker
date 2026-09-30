#!/usr/bin/env python3
"""Fetch current migration totals and append one validated snapshot if changed."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
import uuid

from data_store import ROOT, SOURCE, DICTIONARIES, METRICS, dump_json, iter_snapshots, make_changes, replay, snapshot_path

API = "https://migracje.gov.pl/wp-json/udscmap/v1/"
FIELDS = {
    "decisions": "institution,country,caseType,decisionMarker",
    "applications": "institution,country,caseType",
    "statuses": "institution,status,country",
}
LOOKUPS = {
    "countries": ("country", "name", "code"),
    "caseTypes": ("caseType", "type", None),
    "decisionMarkers": ("decisionMarker", "description", None),
    "statuses": ("status", "status", None),
}


def get_json(endpoint: str, params: dict, insecure: bool):
    url = API + endpoint + "?" + urlencode({**params, "orderBy": uuid.uuid4().hex[:8]})
    context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    with urlopen(url, timeout=45, context=context) as response:
        payload = json.load(response)
    if not isinstance(payload, list) or (payload and not isinstance(payload[0], dict)):
        raise ValueError(f"Unexpected response from {endpoint}: {str(payload)[:180]}")
    return payload


def fetch_metric(metric: str, year: int, insecure: bool):
    fields = FIELDS[metric]
    rows = get_json(f"{metric}/poland", {
        "groupBy": fields, "fields": fields + ",total", "year": year, "limit": -1,
    }, insecure)
    if not rows:
        raise ValueError(f"No {metric} data returned for {year}")
    keys = [name.strip() for name in fields.split(",")]
    canonical = ["country", "institution"] + (["status"] if metric == "statuses" else ["caseType"])
    if metric == "decisions":
        canonical.append("decisionMarker")
    result = defaultdict(int)
    for row in rows:
        if not all(key in row for key in keys) or "total" not in row:
            raise ValueError(f"Malformed {metric} row: {row}")
        value = int(row["total"])
        if value < 0:
            raise ValueError(f"Negative {metric} count")
        result[tuple(int(row[key]) for key in canonical)] += value
    return result


def fetch_institutions(ids: set[int], insecure: bool):
    remote = {}
    ordered = sorted(ids)
    for start in range(0, len(ordered), 60):
        batch = set(ordered[start:start + 60])
        rows = get_json("institution", {
            "include": ",".join(map(str, sorted(batch))), "wpml_language": "pl",
        }, insecure)
        received = {int(row["id"]): row for row in rows}
        if set(received) != batch:
            raise ValueError(f"Institution lookup mismatch: missing {batch - set(received)}, extra {set(received) - batch}")
        for row in rows:
            if not row.get("authorityCode") or not row.get("name"):
                raise ValueError(f"Institution metadata incomplete: {row}")
        remote.update(received)
    return remote


def update_institutions(dictionaries: dict, ids: set[int], insecure: bool):
    existing = {item["id"]: item for item in dictionaries.get("institutions", [])}
    remote = fetch_institutions(set(existing) | ids, insecure)
    for value, row in remote.items():
        item = existing.get(value, {"id": value})
        item["name"] = row["name"]
        item["authorityCode"] = row["authorityCode"]
        item["authorityName"] = row.get("authorityName") or row["authorityCode"]
        item.pop("group", None)
        existing[value] = item
    dictionaries["institutions"] = [existing[key] for key in sorted(existing)]


def refresh_institutions_only(path: Path, insecure: bool):
    dictionaries = json.loads(path.read_text(encoding="utf-8"))
    update_institutions(dictionaries, set(), insecure)
    dump_json(path, dictionaries)
    print(f"Updated authorityCode for {len(dictionaries['institutions'])} institutions")


def fetch_dictionaries(path: Path, current: dict, insecure: bool):
    dictionaries = json.loads(path.read_text(encoding="utf-8"))
    used = {
        "countries": {key[0] for rows in current.values() for key in rows},
        "institutions": {key[1] for rows in current.values() for key in rows},
        "caseTypes": {key[2] for metric in ("decisions", "applications") for key in current[metric]},
        "decisionMarkers": {key[3] for key in current["decisions"]},
        "statuses": {key[2] for key in current["statuses"]},
    }
    update_institutions(dictionaries, used["institutions"], insecure)
    for name, (endpoint, label, code_field) in LOOKUPS.items():
        existing = {item["id"]: item for item in dictionaries.get(name, [])}
        remote = {int(row["id"]): row for row in get_json(endpoint, {"limit": -1}, insecure)}
        if not remote:
            raise ValueError(f"Empty {name} dictionary from source")
        for value in used[name]:
            item = existing.get(value, {"id": value, "name": f"ID {value} · без названия"})
            if value in remote:
                item["name"] = remote[value].get(label) or item["name"]
                if code_field:
                    item["code"] = remote[value].get(code_field) or item.get("code", "")
            existing[value] = item
        dictionaries[name] = [existing[key] for key in sorted(existing)]
    return dictionaries


def fetch_validated(year: int, insecure: bool):
    expected = {}
    for metric in METRICS:
        rows = get_json(f"{metric}/poland", {"groupBy": "", "fields": "total", "year": year}, insecure)
        if len(rows) != 1 or "total" not in rows[0]:
            raise ValueError(f"Invalid {metric} control total")
        expected[metric] = int(rows[0]["total"])
        if expected[metric] < 0:
            raise ValueError(f"Negative {metric} control total")

    current = {metric: fetch_metric(metric, year, insecure) for metric in METRICS}
    for metric in METRICS:
        if sum(current[metric].values()) != expected[metric]:
            raise ValueError(f"{metric} breakdown does not match the source total")
    return current


def retry_source(operation, retry_minutes: float = 15, clock=time.monotonic, pause=time.sleep):
    """Repeat a complete source read until valid or the retry window expires."""
    deadline = clock() + retry_minutes * 60
    attempt = 0
    while True:
        attempt += 1
        try:
            return operation()
        except HTTPError as exc:
            if exc.code not in (408, 429) and exc.code < 500:
                raise
            error = exc
        except (URLError, TimeoutError, ConnectionError, json.JSONDecodeError,
                ValueError, KeyError, TypeError, UnicodeError) as exc:
            error = exc

        remaining = deadline - clock()
        if remaining <= 0:
            raise RuntimeError(f"UdSC response still invalid after {attempt} attempts") from error
        delay = min(remaining, 60, 2 ** min(attempt, 6))
        print(f"UdSC attempt {attempt} failed: {error}; retrying in {delay:g}s", flush=True)
        pause(delay)


def collect(year: int, insecure: bool = False, source: Path = SOURCE, dictionaries: Path = DICTIONARIES,
            retry_minutes: float = 15):
    existing = list(iter_snapshots(source, year))
    previous = replay(existing) if existing else {metric: {} for metric in METRICS}

    def read_source():
        current = fetch_validated(year, insecure)
        changes = {metric: make_changes(previous[metric], current[metric]) for metric in METRICS}
        labels = fetch_dictionaries(dictionaries, current, insecure) if any(changes.values()) else None
        return current, changes, labels

    current, changes, labels = retry_source(read_source, retry_minutes)
    if not any(changes.values()):
        print(f"No data change for {year}")
        return False

    now = datetime.now(timezone.utc)
    timestamp = int(now.timestamp() * 1000) if year == now.year else int(datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc).timestamp() * 1000)
    if existing and timestamp <= existing[-1]["timestamp"]:
        raise ValueError(f"Cannot insert a snapshot before the latest {year} snapshot")
    snapshot = {
        "schemaVersion": 1, "id": f"api-{timestamp}", "timestamp": timestamp, "year": year,
        "totals": {metric: sum(rows.values()) for metric, rows in current.items()},
        "changes": changes,
    }
    path = snapshot_path(source, snapshot)
    if path.exists():
        raise FileExistsError(path)
    dump_json(dictionaries, labels)
    dump_json(path, snapshot)
    replay(iter_snapshots(source, year))
    print(f"Saved {path} with {sum(map(len, changes.values()))} changed rows")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=int(os.environ.get("YEAR") or datetime.now(timezone.utc).year))
    parser.add_argument("--insecure", action="store_true", help="Allow the source site's currently untrusted TLS certificate")
    parser.add_argument("--refresh-institutions", action="store_true", help="Update institution authorityCode without fetching a snapshot")
    parser.add_argument("--retry-minutes", type=float, default=15, help="Retry incomplete source reads for this many minutes (default: 15)")
    args = parser.parse_args()
    if args.retry_minutes < 0:
        parser.error("--retry-minutes must be non-negative")
    if args.refresh_institutions:
        retry_source(lambda: refresh_institutions_only(DICTIONARIES, args.insecure), args.retry_minutes)
    else:
        collect(args.year, args.insecure, retry_minutes=args.retry_minutes)
