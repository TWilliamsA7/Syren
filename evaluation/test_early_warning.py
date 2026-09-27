import math
import os
import tempfile
import xml.etree.ElementTree as ElementTree

from data.test_trace import write_trace
from evaluation.early_warning import (
    GREEN, ORANGE, RED, YELLOW, alert_burden, chance_moments, evaluate_day, find_event, mute_brief,
    mute_takeoff_and_landing, red_reason, summarise,
)
from evaluation.figures import write_report
from shared.flight_state import make_flight_state

LAT, LON = 38.0, -96.0  # inside the app's US area
SPEED_KT, STEP_S = 250.0, 5


def point_at(t, colour, ground=False, causes=None, alt=10000):
    return {"t": t, "flight_id": "TST1", "lat": LAT, "lon": LON, "alt": None if ground else alt,
            "ground": ground, "colour": colour, "causes": causes or ([] if colour == GREEN else [colour])}


def test_takeoff_and_landing_warnings_are_ignored_except_overspeed():
    # on the ground, airborne from 100 s to 1500 s, on the ground again at 1600 s
    warnings = {200: (ORANGE, ["HEADING_OSCILLATION"]),                            # just after takeoff
                300: (ORANGE, ["HEADING_OSCILLATION", "AGGRESSIVE_NEAR_GROUND_SPEED"]),
                800: (ORANGE, ["TELEMETRY_GAP"]),                                  # mid-flight
                1450: (YELLOW, ["prediction SPEED_LOSS"])}                         # just before landing
    timeline = ([point_at(0, GREEN, ground=True)]
                + [point_at(t, warnings.get(t, (GREEN, None))[0], causes=warnings.get(t, (GREEN, None))[1])
                   for t in range(100, 1501, 50)]
                + [point_at(1600, GREEN, ground=True)])
    muted = {p["t"]: p for p in mute_takeoff_and_landing(timeline, takeoff_s=300, landing_s=300)}
    assert muted[200]["colour"] == GREEN
    assert muted[300]["colour"] == ORANGE and muted[300]["causes"] == ["AGGRESSIVE_NEAR_GROUND_SPEED"]
    assert muted[800]["colour"] == ORANGE, "mid-flight warnings stay"
    assert muted[1450]["colour"] == GREEN

    # coming into coverage at cruise altitude isn't a takeoff
    entering = [point_at(t, ORANGE if t == 100 else GREEN, alt=35000) for t in range(0, 1000, 50)]
    assert mute_takeoff_and_landing(entering, 300, 300)[2]["colour"] == ORANGE


def test_brief_orange_is_ignored_but_not_yellow():
    timeline = ([point_at(0, GREEN), point_at(5, ORANGE), point_at(10, GREEN)]      # a 5 s flicker
                + [point_at(t, ORANGE) for t in range(100, 141, 10)] + [point_at(150, GREEN)]  # 50 s
                + [point_at(200, YELLOW), point_at(205, GREEN)])                     # one-report prediction
    muted = {p["t"]: p["colour"] for p in mute_brief(timeline, min_s=15)}
    assert muted[5] == GREEN
    assert all(muted[t] == ORANGE for t in range(100, 141, 10))
    assert muted[200] == YELLOW


def test_red_reason_matches_the_map():
    def state(squawk, emergency):
        return make_flight_state(timestamp=0.0, icao24="a1b2c3", latitude=LAT, longitude=LON,
                                 on_ground=False, source="adsb_icao", origin="history",
                                 squawk=squawk, emergency=emergency)

    assert red_reason(state("7700", "none"), []) == "squawk 7700"
    assert red_reason(state("1200", "lifeguard"), []) == "emergency lifeguard"
    assert red_reason(state("1200", "none"), ["EMERGENCY_SQUAWK"]) == "detector squawk"
    assert red_reason(state("1200", "none"), ["RAPID_DESCENT"]) is None


def test_lead_is_from_the_earliest_warning_before_red():
    timeline = ([point_at(t, GREEN) for t in range(0, 100, 10)]
                + [point_at(100, ORANGE, causes=["RAPID_DESCENT"]), point_at(110, GREEN),
                   point_at(300, YELLOW), point_at(400, RED), point_at(410, RED)])
    event = find_event(timeline)
    assert event["lead_s"] == 300 and event["warning"]["causes"] == ["RAPID_DESCENT"]
    assert event["red_time"] == 400
    assert event["window_causes"] == ["RAPID_DESCENT", "yellow"], "every kind of warning in the window"


def test_chance_moments_use_the_same_window():
    # a normal flight checked every 5 minutes: warned at 600 s (orange at 500 s), not at 0 or 1800 s
    timeline = [point_at(t, ORANGE if t == 500 else GREEN, causes=["TELEMETRY_GAP"] if t == 500 else None)
                for t in range(0, 1900, 100)]
    chance = chance_moments(timeline, every_s=600)
    assert chance["moments"] == 4, "at 0, 600, 1200 and 1800 s"
    assert chance["warned"] == 2 and chance["by_cause"] == {"TELEMETRY_GAP": 2}, "600 and 1200 s"


def test_lift_compares_emergencies_with_chance():
    event = {"lead_s": 120.0, "warning": {"colour": ORANGE, "causes": ["RAPID_DESCENT"]},
             "window_causes": ["RAPID_DESCENT"]}
    day = {"date": "2026-09-24", "aircraft": 10, "excluded": {}, "events": [event, {**event, "lead_s": None,
                                                                                   "warning": None, "window_causes": []}],
           "sample": {"aircraft": 5, "flagged": 1, "airborne_hours": 5, "onsets": 2, "by_cause": {}},
           "chance": {"moments": 100, "warned": 25, "by_cause": {"RAPID_DESCENT": 5, "TELEMETRY_GAP": 20}}}
    summary = summarise([day])
    assert summary["warned_pct"] == 50.0 and summary["chance_warned_pct"] == 25.0 and summary["lift"] == 2.0
    assert summary["warned_pct_95"] == [9.5, 90.5], "1 of 2 is a very uncertain 50%"
    rows = {row["cause"]: row for row in summary["by_warning"]}
    assert rows["RAPID_DESCENT"]["lift"] == 10.0, "50% before emergencies vs 5% of ordinary moments"
    assert rows["TELEMETRY_GAP"]["before_emergencies_pct"] == 0.0 and rows["TELEMETRY_GAP"]["lift"] == 0.0


def test_what_does_not_count():
    blip = [point_at(0, GREEN), point_at(10, RED), point_at(20, GREEN), point_at(30, GREEN)]
    assert find_event(blip) is None, "one red report is a blip, not an emergency"

    on_ground = [point_at(0, RED, ground=True), point_at(10, RED, ground=True)]
    assert find_event(on_ground) is None, "squawking on the ground isn't an airborne emergency"

    too_early = [point_at(0, ORANGE)] + [point_at(t, GREEN) for t in range(60, 1500, 60)] \
        + [point_at(1500, RED), point_at(1510, RED)]
    assert find_event(too_early)["warning"] is None, "a warning over 20 minutes before doesn't count"

    after_gap = [point_at(0, ORANGE), point_at(400, GREEN), point_at(410, RED), point_at(420, RED)]
    assert find_event(after_gap)["warning"] is None, "a warning before a 5+ minute gap doesn't count"


def test_alert_burden_counts_warning_onsets():
    timeline = [point_at(0, GREEN), point_at(10, ORANGE, causes=["TELEMETRY_GAP"]), point_at(20, ORANGE),
                point_at(30, GREEN), point_at(40, YELLOW, causes=["prediction SPEED_LOSS"]),
                point_at(50, GREEN, ground=True)]
    burden = alert_burden(timeline)
    assert burden["onsets"] == 2
    assert burden["by_cause"] == {"TELEMETRY_GAP": 1, "prediction SPEED_LOSS": 1}
    assert burden["airborne_s"] == 40, "the last report is on the ground"


# One aircraft flying east at a steady 250 kt, so position, speed and track agree.
def flight(seconds, altitude_at, rate_at, details_at):
    lon_step = SPEED_KT * STEP_S / 3600 / (60 * math.cos(math.radians(LAT)))
    points = []
    for i, t in enumerate(seconds):
        points.append([t, LAT, LON + i * lon_step, altitude_at(t), SPEED_KT, 90.0, 0, rate_at(t),
                       details_at(t), "adsb_icao", None, None, None, None])
    return points


def test_evaluate_a_day_end_to_end():
    start = 16 * 3600
    seconds = list(range(start, start + 900, STEP_S))
    descent_starts, squawk_at = start + 600, start + 780

    def altitude(t):
        return 35000 if t < descent_starts else 35000 - (t - descent_starts) * 4500 / 60

    with tempfile.TemporaryDirectory() as root:
        day = os.path.join(root, "fixed", "2026-09-24")
        write_trace(day, "a00001", flight(
            seconds, altitude, lambda t: 0 if t < descent_starts else -4480,
            lambda t: ({"flight": "EMG123", "squawk": "1234", "category": "A3"} if t == start
                       else {"squawk": "7700", "emergency": "general"} if t == squawk_at else None)),
            t="B738")
        write_trace(day, "a00002", flight(
            seconds, lambda t: 33000, lambda t: 0,
            lambda t: {"flight": "NRM456", "squawk": "2345"} if t == start else None))
        write_trace(day, "a00003", flight(
            seconds, lambda t: 9000, lambda t: 0,
            lambda t: {"flight": "MED789", "squawk": "3456", "emergency": "lifeguard"} if t == start else None))

        result = evaluate_day("2026-09-24", root=root, sample_per_mille=1000)
        assert result["aircraft"] == 3
        assert result["excluded"] == {"lifeguard": 1, "brief_or_on_ground": 0}
        assert result["sample"]["aircraft"] == 1, "only the aircraft with no emergency is a sample"

        [event] = result["events"]
        assert event["flight_id"] == "EMG123" and event["reason"] == "squawk 7700"
        assert event["red_time"] == 1790208000.0 + squawk_at
        assert event["warning"] is not None and event["warning"]["colour"] in (YELLOW, ORANGE)
        assert 0 < event["lead_s"] <= squawk_at - descent_starts + STEP_S, event

        summary = summarise([result])
        assert summary["events"] == 1 and summary["warned"] == 1 and summary["warned_pct"] == 100.0

        out = os.path.join(root, "evaluation")
        write_report([result], summary, out)
        assert os.path.exists(os.path.join(out, "report.html"))
        ElementTree.parse(os.path.join(out, "lead_times.svg"))  # raises if the chart isn't valid SVG
        svg = os.path.join(out, "events", "2026-09-24_a00001.svg")
        colours = {line.get("stroke") for line in ElementTree.parse(svg).iter("{http://www.w3.org/2000/svg}line")}
        assert "#F43F5E" in colours and ({"#FB923C", "#FACC15"} & colours), "the image shows warning then red"


if __name__ == "__main__":
    test_red_reason_matches_the_map()
    test_lead_is_from_the_earliest_warning_before_red()
    test_chance_moments_use_the_same_window()
    test_takeoff_and_landing_warnings_are_ignored_except_overspeed()
    test_brief_orange_is_ignored_but_not_yellow()
    test_lift_compares_emergencies_with_chance()
    test_what_does_not_count()
    test_alert_burden_counts_warning_onsets()
    test_evaluate_a_day_end_to_end()
    print("all tests passed")
