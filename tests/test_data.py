import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_store import METRICS, SOURCE, iter_snapshots, replay  # noqa: E402


class SnapshotIntegrityTests(unittest.TestCase):
    def test_build_version_matches_every_published_pack(self):
        manifest = json.loads((ROOT / "dist/data/manifest.json").read_text(encoding="utf-8"))
        build_id = manifest["buildId"]
        self.assertRegex(build_id, r"^[0-9a-f]{24}$")
        for year, metadata in manifest["years"].items():
            paths = sorted((ROOT / "dist/data" / year).glob("*.json"))
            self.assertEqual(set(metadata["packHashes"]), {path.stem for path in paths})
            for path in paths:
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 metadata["packHashes"][path.stem])
        index = (ROOT / "dist/index.html").read_text(encoding="utf-8")
        app = (ROOT / "dist/web/app.js").read_text(encoding="utf-8")
        self.assertIn(f"./web/app.js?v={build_id}", index)
        self.assertIn(f"./web/styles.css?v={build_id}", index)
        self.assertIn(f"./data.js?v={build_id}", app)
        self.assertIn(f"const APP_BUILD_ID = '{build_id}'", app)

    def test_transient_source_outages_are_removed_from_published_series(self):
        manifest = json.loads((ROOT / "dist/data/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["years"]["2024"]["omittedSourceMetrics"], [
            {"id": 570, "date": "2024-04-21", "metrics": ["statuses"]},
            {"id": 577, "date": "2024-04-28", "metrics": ["statuses"]},
            {"id": 581, "date": "2024-05-02", "metrics": ["statuses"]},
        ])
        self.assertEqual(manifest["years"]["2025"]["omittedSourceMetrics"], [
            {"id": 919, "date": "2025-05-01", "metrics": list(METRICS)},
        ])
        points_2024 = manifest["years"]["2024"]["snapshots"]
        for day in ("2024-04-21", "2024-04-28", "2024-05-02"):
            index = next(i for i, point in enumerate(points_2024) if point["date"] == day)
            self.assertEqual(points_2024[index]["totals"]["statuses"],
                             points_2024[index - 1]["totals"]["statuses"])
            self.assertGreater(points_2024[index]["totals"]["decisions"],
                               points_2024[index - 1]["totals"]["decisions"])
        self.assertNotIn("2025-05-01", {point["date"] for point in
                                          manifest["years"]["2025"]["snapshots"]})
        self.assertNotIn("omittedSourceMetrics", manifest["years"]["2026"])
        self.assertIn("2026-02-26", {point["date"] for point in
                                        manifest["years"]["2026"]["snapshots"]})

    def test_every_snapshot_reconstructs_its_stored_totals(self):
        years = sorted(int(path.name) for path in SOURCE.iterdir() if path.is_dir())
        self.assertTrue(years)
        for year in years:
            states = {metric: {} for metric in METRICS}
            for snapshot in iter_snapshots(SOURCE, year):
                self.assertEqual(set(snapshot["changes"]), set(METRICS))
                self.assertEqual(set(snapshot["totals"]), set(METRICS))
                for metric in METRICS:
                    for *key, change in snapshot["changes"][metric]:
                        key = tuple(key)
                        states[metric][key] = states[metric].get(key, 0) + change
                        self.assertGreaterEqual(states[metric][key], 0)
                        if states[metric][key] == 0:
                            del states[metric][key]
                    self.assertEqual(sum(states[metric].values()), snapshot["totals"][metric],
                                     (year, snapshot["id"], metric))
            self.assertEqual(replay(iter_snapshots(SOURCE, year)), states)

    def test_published_totals_match_filtered_rows(self):
        manifest = json.loads((ROOT / "dist/data/manifest.json").read_text(encoding="utf-8"))
        self.assertIn("statuses", manifest["dictionaries"])
        for year, meta in manifest["years"].items():
            pack = json.loads((ROOT / f"dist/data/{year}/all.json").read_text(encoding="utf-8"))
            self.assertEqual(set(pack), set(METRICS))
            for metric in METRICS:
                running = 0
                by_index = {}
                for index, *dims, change in pack[metric]:
                    by_index[index] = by_index.get(index, 0) + change
                for index, point in enumerate(meta["snapshots"]):
                    self.assertEqual(set(point["totals"]), set(METRICS))
                    running += by_index.get(index, 0)
                    self.assertEqual(running, point["totals"][metric], (year, index, metric))

    def test_publication_uses_metric_specific_institution_scope(self):
        source_labels = json.loads((ROOT / "data/dictionaries.json").read_text(encoding="utf-8"))
        voivodes = {item["id"] for item in source_labels["institutions"]
                    if item["name"].startswith("Wojewoda ")}
        self.assertEqual(len(voivodes), 16)
        decision_institutions = voivodes | {810}
        manifest = json.loads((ROOT / "dist/data/manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn(5, {item["id"] for item in manifest["dictionaries"]["caseTypes"]})
        self.assertEqual(set(manifest["institutionsByMetric"]["decisions"]), decision_institutions)
        institutions = {item["id"]: item for item in manifest["dictionaries"]["institutions"]}
        self.assertEqual(institutions[3201]["authorityCode"], "PSG")
        self.assertEqual(institutions[1239]["authorityCode"], "OSG")
        self.assertEqual(institutions[810]["authorityCode"], "MIN")
        self.assertEqual(institutions[882]["authorityCode"], "WOJ")
        self.assertNotIn(867, manifest["institutionsByMetric"]["decisions"])
        self.assertIn(3201, manifest["institutionsByMetric"]["applications"])
        self.assertIn(1239, manifest["institutionsByMetric"]["applications"])
        self.assertNotIn(3201, manifest["institutionsByMetric"]["decisions"])
        for path in (ROOT / "dist/data").glob("*/*.json"):
            pack = json.loads(path.read_text(encoding="utf-8"))
            for metric, rows in pack.items():
                for row in rows:
                    self.assertIn(row[1], manifest["institutionsByMetric"][metric], (path, metric))
                    if metric != "applications":
                        self.assertIn(row[1], decision_institutions, (path, metric))
                    if metric in ("decisions", "applications"):
                        self.assertNotEqual(row[2], 5, (path, metric))

        for year in manifest["years"]:
            source = replay(iter_snapshots(SOURCE, int(year)))
            pack = json.loads((ROOT / f"dist/data/{year}/all.json").read_text(encoding="utf-8"))
            for metric in METRICS:
                expected = sum(value for (country, institution, dimension, *rest), value in source[metric].items()
                               if (metric == "applications" or institution in decision_institutions)
                               and (metric == "statuses" or dimension != 5))
                self.assertEqual(sum(row[-1] for row in pack[metric]), expected, (year, metric))
            protection_source = sum(value for (country, institution, case_type), value in source["applications"].items()
                                    if case_type == 4)
            protection_published = sum(row[-1] for row in pack["applications"] if row[2] == 4)
            self.assertEqual(protection_published, protection_source, year)


if __name__ == "__main__":
    unittest.main()
