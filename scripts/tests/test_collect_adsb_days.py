import gzip
import json
import tarfile
import tempfile
import unittest
from datetime import date
from pathlib import Path

from scripts.collect_adsb_days import (
    ACTIVE_EMERGENCY_MARKER,
    EMERGENCY_SQUAWK_MARKER,
    inspect_trace,
    scan_archive,
)


def trace_payload(points, icao="abcdef"):
    raw = json.dumps({
        "icao": icao,
        "r": "N123AB",
        "t": "B738",
        "flight": "TEST123",
        "timestamp": 1735689600,
        "trace": points,
    }).encode()
    return gzip.compress(raw)


class CollectDaysTests(unittest.TestCase):
    def test_detects_emergency_squawk_and_emergency_field(self):
        parsed = inspect_trace(trace_payload([
            [0, 38.0, -77.0, 10000, 250, 90, 0, 0, {"squawk": "1200", "emergency": "none"}],
            [10, 38.1, -77.0, 9000, 250, 90, 0, -1000, {"squawk": "7700", "emergency": "general"}],
            [20, 38.2, -77.0, 8000, 250, 90, 0, -1000, None],
        ]))
        self.assertEqual(
            [(signal["signal"], signal["value"]) for signal in parsed["signals"]],
            [("emergency_field", "general"), ("emergency_squawk", "7700")],
        )

    def test_non_emergency_trace_returns_empty_signal_list(self):
        parsed = inspect_trace(trace_payload([
            [0, 38.0, -77.0, 10000, 250, 90, 0, 0, {"squawk": "1200", "emergency": "none"}],
        ]))
        self.assertEqual(parsed["signals"], [])

    def test_fast_marker_skips_no_emergency_and_detects_declared_signals(self):
        ordinary = b'{"trace":[{"emergency":"none","squawk":"1200"}]}'
        emergency = b'{"trace":[{"emergency":"general","squawk":"7700"}]}'
        self.assertFalse(ACTIVE_EMERGENCY_MARKER.search(ordinary))
        self.assertFalse(EMERGENCY_SQUAWK_MARKER.search(ordinary))
        self.assertTrue(ACTIVE_EMERGENCY_MARKER.search(emergency))
        self.assertTrue(EMERGENCY_SQUAWK_MARKER.search(emergency))

    def test_streams_tar_and_keeps_positive_plus_deterministic_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "daily.tar"
            with tarfile.open(archive_path, "w") as archive:
                payloads = {
                    "traces/ab/trace_full_abcdef.json": trace_payload([
                        [0, 38, -77, 10000, 200, 90, 0, 0, {"squawk": "7700"}],
                    ]),
                    "traces/12/trace_full_123412.json": trace_payload([
                        [0, 39, -76, 10000, 200, 90, 0, 0, {"squawk": "1200"}],
                    ], "123412"),
                    "traces/34/trace_full_123434.json": trace_payload([
                        [0, 40, -75, 10000, 200, 90, 0, 0, None],
                    ], "123434"),
                }
                for name, payload in payloads.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, __import__("io").BytesIO(payload))

            report = scan_archive(archive_path, date(2025, 1, 1), root / "out", 1, ["https://example/archive.tar"])
            self.assertEqual(report["aircraft_trace_files_scanned"], 3)
            self.assertEqual(report["aircraft_with_explicit_emergency_signal"], 1)
            self.assertEqual(report["control_traces_retained"], 1)
            self.assertTrue((root / "out" / "traces/2025-01-01/emergency/abcdef.json.gz").exists())
            self.assertEqual(report["controls"][0]["label"], "unlabeled_no_observed_declaration")


if __name__ == "__main__":
    unittest.main()
