import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect import collect, fetch_validated, reject_incomplete, retry_source  # noqa: E402


class SourceRetryTests(unittest.TestCase):
    def test_rejects_catastrophic_drop_before_saving(self):
        previous = {"decisions": {(241, 810, 4, 6): 5000},
                    "applications": {(241, 810, 4): 7000},
                    "statuses": {(241, 810, 2): 9000}}
        incomplete = {**previous, "statuses": {(241, 810, 2): 1200}}
        with self.assertRaisesRegex(ValueError, "statuses total fell"):
            reject_incomplete(previous, incomplete)
        reject_incomplete(previous, {**previous, "applications": {(241, 810, 4): 5000}})

    def test_retries_transient_invalid_response(self):
        elapsed = [0.0]
        delays = []
        attempts = [0]

        def operation():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("incomplete response")
            return "valid"

        def pause(delay):
            delays.append(delay)
            elapsed[0] += delay

        result = retry_source(operation, 1, clock=lambda: elapsed[0], pause=pause)
        self.assertEqual(result, "valid")
        self.assertEqual(attempts[0], 3)
        self.assertEqual(delays, [2, 4])

    def test_stops_after_retry_window_and_keeps_snapshot_untouched(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "snapshots"
            labels = Path(directory) / "dictionaries.json"
            labels.write_text(json.dumps({"institutions": []}), encoding="utf-8")
            with patch("collect.fetch_validated", side_effect=ValueError("bad source")):
                with self.assertRaisesRegex(RuntimeError, "still invalid"):
                    collect(2026, source=source, dictionaries=labels, retry_minutes=0)
            self.assertFalse(source.exists())
            self.assertEqual(json.loads(labels.read_text()), {"institutions": []})

    def test_permanent_http_error_is_not_retried(self):
        attempts = [0]
        error = HTTPError("https://migracje.gov.pl/", 404, "missing", {}, None)

        def operation():
            attempts[0] += 1
            raise error

        try:
            with self.assertRaises(HTTPError):
                retry_source(operation, 1)
        finally:
            error.close()
        self.assertEqual(attempts[0], 1)

    def test_retries_server_error(self):
        attempts = [0]
        error = HTTPError("https://migracje.gov.pl/", 503, "unavailable", {}, None)

        def operation():
            attempts[0] += 1
            if attempts[0] == 1:
                raise error
            return "valid"

        elapsed = [0]

        def pause(delay):
            elapsed[0] += delay

        try:
            result = retry_source(operation, 1, clock=lambda: elapsed[0], pause=pause)
        finally:
            error.close()
        self.assertEqual(result, "valid")
        self.assertEqual(attempts[0], 2)

    def test_saves_snapshot_only_after_complete_source_read(self):
        current = {
            "decisions": {(241, 810, 4, 6): 3},
            "applications": {(241, 810, 4): 5},
            "statuses": {(241, 810, 2): 7},
        }
        with TemporaryDirectory() as directory:
            source = Path(directory) / "snapshots"
            labels = Path(directory) / "dictionaries.json"
            labels.write_text("{}", encoding="utf-8")
            with patch("collect.fetch_validated", return_value=current), \
                    patch("collect.fetch_dictionaries", return_value={"ready": True}):
                self.assertTrue(collect(2026, source=source, dictionaries=labels, retry_minutes=0))
            files = list(source.glob("2026/*.json"))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text())["totals"],
                             {"decisions": 3, "applications": 5, "statuses": 7})
            self.assertEqual(json.loads(labels.read_text()), {"ready": True})

    def test_rejects_inconsistent_application_total(self):
        totals = {"decisions": 3, "applications": 5, "statuses": 7}

        def get_json(endpoint, params, insecure):
            return [{"total": totals[endpoint.split("/")[0]]}]

        def fetch_metric(metric, year, insecure):
            count = totals[metric] - (1 if metric == "applications" else 0)
            return {(241, 810, 4): count}

        with patch("collect.get_json", side_effect=get_json), patch("collect.fetch_metric", side_effect=fetch_metric):
            with self.assertRaisesRegex(ValueError, "applications breakdown"):
                fetch_validated(2026, False)


if __name__ == "__main__":
    unittest.main()
