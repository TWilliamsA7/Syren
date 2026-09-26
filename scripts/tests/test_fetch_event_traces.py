import unittest

from scripts.fetch_event_traces import TraceError, trace_points, trace_url


class FetchTraceTests(unittest.TestCase):
    def test_trace_url_uses_utc_date_and_last_two_hex_digits(self):
        self.assertEqual(
            trace_url("2026-03-23", "C06800"),
            "https://adsb.lol/globe_history/2026/03/23/traces/00/trace_full_c06800.json",
        )

    def test_normalizes_trace_tuple_and_preserves_ground(self):
        trace = {
            "icao": "c06800",
            "r": "C-GNJZ",
            "t": "CRJ9",
            "timestamp": 1774236900.0,
            "trace": [
                [0.0, 40.7, -73.8, "ground", 0.0, 270.0, 0, 0, None, "adsb_icao", None, None, None, None],
                [10.0, 40.71, -73.81, 500, 130.0, 270.0, 2, 1000, {"flight": "JZA646", "squawk": "1200"}, "adsb_icao", 525, 1024, 128, 1.0],
            ],
        }
        rows = list(trace_points(trace, "c06800"))
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["on_ground"])
        self.assertIsNone(rows[0]["altitude_baro_ft"])
        self.assertEqual(rows[1]["vertical_rate_geom_fpm"], 1024.0)
        self.assertEqual(rows[1]["extra"], {"flight": "JZA646", "squawk": "1200"})
        self.assertEqual(rows[1]["flight_leg_index"], 1)
        self.assertEqual(rows[1]["callsign"], "JZA646")

    def test_rejects_identity_mismatch(self):
        with self.assertRaises(TraceError):
            list(trace_points({"icao": "abcdef", "timestamp": 1, "trace": []}, "c06800"))


if __name__ == "__main__":
    unittest.main()
