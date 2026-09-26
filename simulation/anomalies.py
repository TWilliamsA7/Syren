import math

EMERGENCY_FOR_SQUAWK = {"7500": "unlawful", "7600": "nordo", "7700": "general"}


def rapid_descent(ac, elapsed_s, params):
    """Descent that steepens to vertical_rate_fpm over ramp_s, ignoring the autopilot."""
    peak_fpm = params.get("vertical_rate_fpm", -6000)
    ramp_s = params.get("ramp_s", 60)
    ac.target_vertical_rate_fpm = peak_fpm * min(1.0, elapsed_s / ramp_s)


def erratic_altitude(ac, elapsed_s, params):
    """Irregular climbs and descents, as in an upset or a control problem."""
    amplitude_fpm = params.get("amplitude_fpm", 3000)
    wave = (0.6 * math.sin(2 * math.pi * elapsed_s / 120)
            + 0.4 * math.sin(2 * math.pi * elapsed_s / 45 + 1))
    ac.target_vertical_rate_fpm = amplitude_fpm * wave


def speed_loss(ac, elapsed_s, params):
    """Airspeed decays toward speed_kts, e.g. from a stall or an engine problem."""
    ac.target_speed_kts = params.get("speed_kts", 200)


def low_altitude_overspeed(ac, elapsed_s, params):
    """Fast and low: speed_kts at or below altitude_ft, well over the 250 kt limit."""
    ac.target_speed_kts = params.get("speed_kts", 350)
    ac.selected_altitude_ft = params.get("altitude_ft", 2000)
    ac.descent_rate_fpm = params.get("descent_rate_fpm", 3000)


def route_deviation(ac, elapsed_s, params):
    """Fly offset_deg off the planned track, reached gradually over ramp_s."""
    if ac.target_track_deg is None:
        return
    offset_deg = params.get("offset_deg", 45)
    ramp_s = params.get("ramp_s", 0)
    fraction = min(1.0, elapsed_s / ramp_s) if ramp_s else 1.0
    ac.target_track_deg = (ac.target_track_deg + offset_deg * fraction) % 360


def squawk(ac, elapsed_s, params):
    """Transponder emergency code: 7700 emergency, 7600 radio failure, 7500 hijack."""
    code = params.get("code", "7700")
    ac.squawk = code
    ac.emergency = EMERGENCY_FOR_SQUAWK.get(code, "general")


def signal_loss(ac, elapsed_s, params):
    """Transponder goes silent; the aircraft keeps flying but stops reporting."""
    ac.transponder_on = False


def engine_failure(ac, elapsed_s, params):
    """Loses thrust: slows, drifts down, and declares 7700 only after declare_after_s."""
    ac.target_speed_kts = params.get("speed_kts", 260)
    if ac.selected_altitude_ft is not None:
        ac.selected_altitude_ft = min(ac.selected_altitude_ft, params.get("drift_down_ft", 20000))
    ac.descent_rate_fpm = params.get("descent_rate_fpm", 1200)
    ac.climb_rate_fpm = 0
    if elapsed_s >= params.get("declare_after_s", 90):
        squawk(ac, elapsed_s, {"code": "7700"})


def hijack(ac, elapsed_s, params):
    """Unlawful interference: 7500, a sharp turn off the route, an altitude change,
    and optionally the transponder switched off after transponder_off_after_s."""
    route_deviation(ac, elapsed_s, {"offset_deg": params.get("offset_deg", 120)})
    if "altitude_ft" in params:
        ac.selected_altitude_ft = params["altitude_ft"]
    if params.get("squawk_7500", True):
        squawk(ac, elapsed_s, {"code": "7500"})
    off_after_s = params.get("transponder_off_after_s")
    if off_after_s is not None and elapsed_s >= off_after_s:
        signal_loss(ac, elapsed_s, params)


ANOMALIES = {
    "rapid_descent": rapid_descent,
    "erratic_altitude": erratic_altitude,
    "speed_loss": speed_loss,
    "low_altitude_overspeed": low_altitude_overspeed,
    "route_deviation": route_deviation,
    "squawk": squawk,
    "signal_loss": signal_loss,
    "engine_failure": engine_failure,
    "hijack": hijack,
}


def check_anomalies(anomalies):
    """Fail early on a scenario that names an anomaly type that doesn't exist."""
    for anomaly in anomalies:
        if anomaly["type"] not in ANOMALIES:
            raise ValueError(
                f"unknown anomaly type {anomaly['type']!r}, expected one of {sorted(ANOMALIES)}")


def apply_anomalies(anomalies, t, aircraft_by_icao24):
    """Apply every anomaly whose window contains t (seconds since scenario start)."""
    for anomaly in anomalies:
        ac = aircraft_by_icao24.get(anomaly["icao24"])
        if ac is None:
            continue
        elapsed_s = t - anomaly["start_s"]
        if 0 <= elapsed_s < anomaly["duration_s"]:
            ANOMALIES[anomaly["type"]](ac, elapsed_s, anomaly.get("params", {}))


def ground_truth(scenario):
    """What was injected and when, in absolute time. Never give this to the detector."""
    start_time = scenario["start_time"]
    return [
        {
            "icao24": anomaly["icao24"],
            "type": anomaly["type"],
            "start_time": start_time + anomaly["start_s"],
            "end_time": start_time + anomaly["start_s"] + anomaly["duration_s"],
            "params": anomaly.get("params", {}),
        }
        for anomaly in scenario.get("anomalies", [])
    ]