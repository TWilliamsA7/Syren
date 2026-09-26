import argparse
import gzip
import json
import os
import random

from simulation.anomalies import ANOMALIES, ground_truth
from simulation.flight_plan import AIRPORTS, distance_and_bearing
from simulation.run import depart, run_scenario

START_TIME = 1790400000
TYPE_CODES = ["A319", "A320", "A321", "A20N", "A21N", "B737", "B738", "B38M", "B39M",
              "B752", "E75L", "CRJ9"]
HEAVY_TYPES = {"B752"}
RESERVED_SQUAWKS = {"0000", "1200", "7500", "7600", "7700", "7777"}

# Where in a flight each anomaly can start, from the flight's phase and altitude then.
ELIGIBLE = {
    "rapid_descent": lambda phase, alt: phase in ("climb", "cruise") and alt > 20000,
    "erratic_altitude": lambda phase, alt: phase in ("climb", "cruise", "descent") and alt > 12000,
    "speed_loss": lambda phase, alt: phase in ("climb", "cruise") and alt > 15000,
    "low_altitude_overspeed": lambda phase, alt: phase in ("takeoff", "descent") and 3000 < alt < 10000,
    "route_deviation": lambda phase, alt: phase in ("climb", "cruise", "descent") and alt > 10000,
    "squawk": lambda phase, alt: phase != "landed",
    "signal_loss": lambda phase, alt: phase != "landed" and alt > 3000,
    "engine_failure": lambda phase, alt: phase in ("climb", "cruise") and alt > 20000,
    "hijack": lambda phase, alt: phase in ("climb", "cruise") and alt > 15000,
}


def _random_squawk(rng):
    while True:
        code = "".join(rng.choice("01234567") for _ in range(4))
        if code not in RESERVED_SQUAWKS:
            return code


def _random_flight(rng, index, depart_at_s):
    origin, destination = rng.sample(sorted(AIRPORTS), 2)
    a, b = AIRPORTS[origin], AIRPORTS[destination]
    distance_nm, bearing = distance_and_bearing(a.latitude, a.longitude, b.latitude, b.longitude)

    # Short hops fly lower; semicircular rule: odd thousands eastbound, even westbound.
    ceiling = int(min(39000, 12000 + distance_nm * 150) / 1000)
    thousands = rng.randint(max(12, ceiling - 6), ceiling)
    if (thousands % 2 == 1) != (bearing < 180):
        thousands -= 1

    type_code = rng.choice(TYPE_CODES)
    return {
        "icao24": f"~{0x800000 + index:06x}",
        "callsign": f"SYN{index:04d}",
        "type_code": type_code,
        "category": "A4" if type_code in HEAVY_TYPES else "A3",
        "squawk": _random_squawk(rng),
        "origin": origin,
        "destination": destination,
        "cruise_altitude_ft": thousands * 1000,
        "cruise_speed_kts": round(rng.uniform(420, 480)),
        "climb_rate_fpm": round(rng.uniform(1500, 2500)),
        "descent_rate_fpm": round(rng.uniform(1500, 2800)),
        "depart_at_s": depart_at_s,
    }


def flight_timeline(flight):
    """(phase, altitude) for each second after departure, if nothing goes wrong."""
    plan, ac = depart(flight)
    timeline = []
    while not plan.finished and len(timeline) < 4 * 3600:
        plan.update(ac)
        timeline.append((plan.phase, ac.altitude_ft))
        ac.step(1.0)
    return timeline


def _random_anomaly(kind, rng, altitude_ft, remaining_s):
    """A random (duration_s, params) for one anomaly of this kind."""
    sign = rng.choice([-1, 1])
    if kind == "rapid_descent":
        rate = rng.uniform(-8000, -4000)
        ramp_s = rng.uniform(20, 90)
        longest_s = ramp_s / 2 + (altitude_ft - 8000) * 60 / -rate  # stay above ~8000 ft
        return min(rng.uniform(60, 180), longest_s), {
            "vertical_rate_fpm": round(rate), "ramp_s": round(ramp_s)}
    if kind == "erratic_altitude":
        return rng.uniform(120, 300), {"amplitude_fpm": round(rng.uniform(2000, 4000))}
    if kind == "speed_loss":
        return rng.uniform(120, 300), {"speed_kts": round(rng.uniform(180, 260))}
    if kind == "low_altitude_overspeed":
        return rng.uniform(120, 300), {
            "speed_kts": round(rng.uniform(320, 380)),
            "altitude_ft": rng.randrange(1500, 3001, 100)}
    if kind == "route_deviation":
        return rng.uniform(120, 600), {
            "offset_deg": sign * round(rng.uniform(30, 90)), "ramp_s": round(rng.uniform(0, 120))}
    if kind == "squawk":
        return rng.uniform(300, 1200), {"code": rng.choice(["7500", "7600", "7700"])}
    if kind == "signal_loss":
        return rng.uniform(60, 300), {}
    if kind == "engine_failure":
        return remaining_s + 1800, {
            "speed_kts": round(rng.uniform(240, 280)),
            "drift_down_ft": rng.randrange(15000, 22001, 1000),
            "descent_rate_fpm": round(rng.uniform(800, 1500)),
            "declare_after_s": round(rng.uniform(30, 180))}
    if kind == "hijack":
        params = {"offset_deg": sign * round(rng.uniform(90, 150)),
                  "squawk_7500": rng.random() < 0.7}
        if rng.random() < 0.6:
            params["altitude_ft"] = rng.randrange(8000, 20001, 1000)
        if rng.random() < 0.5:
            params["transponder_off_after_s"] = round(rng.uniform(60, 300))
        return remaining_s + 1800, params
    raise ValueError(f"no parameters defined for anomaly type {kind!r}")


def _place_anomaly(rng, flight, kind, timeline):
    """Pick a random eligible moment in the flight for this anomaly, or None if there is none."""
    starts = [i for i, (phase, alt) in enumerate(timeline) if ELIGIBLE[kind](phase, alt)]
    if not starts:
        return None
    offset_s = rng.choice(starts)
    duration_s, params = _random_anomaly(kind, rng, timeline[offset_s][1], len(timeline) - offset_s)
    return {
        "icao24": flight["icao24"],
        "type": kind,
        "start_s": flight["depart_at_s"] + offset_s,
        "duration_s": round(duration_s),
        "params": params,
    }


def make_scenario(seed, num_flights, anomaly_rate, depart_window_s=1800, noise=True):
    """A random scenario: num_flights flights, each with one anomaly with probability anomaly_rate."""
    rng = random.Random(seed)
    flights = [_random_flight(rng, i, round(rng.uniform(0, depart_window_s)))
               for i in range(num_flights)]

    anomalies = []
    for flight in flights:
        if rng.random() >= anomaly_rate:
            continue
        timeline = flight_timeline(flight)
        kinds = sorted(ANOMALIES)
        rng.shuffle(kinds)
        for kind in kinds:
            anomaly = _place_anomaly(rng, flight, kind, timeline)
            if anomaly is not None:
                anomalies.append(anomaly)
                break

    scenario = {
        "name": f"generated_seed{seed}",
        "description": f"{num_flights} random flights, about {anomaly_rate:.0%} with one anomaly.",
        "start_time": START_TIME,
        "duration_s": depart_window_s + 4500,
        "dt": 1.0,
        "flights": flights,
        "anomalies": anomalies,
    }
    if noise:
        scenario["noise_seed"] = seed
    return scenario


def label_for(state, windows):
    """The anomaly active for this aircraft when this state was reported, else "normal"."""
    reported_at = state["timestamp"] + (state["status"]["seen_age_s"] or 0)
    window = windows.get(state["icao24"])
    active = window is not None and window["start_time"] <= reported_at < window["end_time"]
    return {
        "icao24": state["icao24"],
        "timestamp": state["timestamp"],
        "label": window["type"] if active else "normal",
    }


def write_dataset(out_dir, scenario):
    """Run the scenario and write the dataset files; return how many states got each label."""
    os.makedirs(out_dir, exist_ok=True)
    truth = ground_truth(scenario)
    windows = {entry["icao24"]: entry for entry in truth}  # at most one anomaly per flight

    with open(os.path.join(out_dir, "scenario.json"), "w") as f:
        json.dump(scenario, f, indent=2)
    with open(os.path.join(out_dir, "truth.json"), "w") as f:
        json.dump(truth, f, indent=2)

    counts = {}
    with gzip.open(os.path.join(out_dir, "states.jsonl.gz"), "wt") as states_file, \
            gzip.open(os.path.join(out_dir, "labels.jsonl.gz"), "wt") as labels_file:
        for state in run_scenario(scenario):
            label = label_for(state, windows)
            states_file.write(json.dumps(state) + "\n")
            labels_file.write(json.dumps(label) + "\n")
            counts[label["label"]] = counts.get(label["label"], 0) + 1
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a labelled dataset of simulated flights.")
    parser.add_argument("out_dir", help="folder to write the dataset into")
    parser.add_argument("--flights", type=int, default=100, help="number of flights (default 100)")
    parser.add_argument("--anomaly-rate", type=float, default=0.3,
                        help="fraction of flights that get one anomaly (default 0.3)")
    parser.add_argument("--seed", type=int, default=0, help="same seed, same dataset (default 0)")
    parser.add_argument("--depart-window", type=int, default=1800,
                        help="flights depart within this many seconds (default 1800)")
    parser.add_argument("--no-noise", action="store_true", help="no sensor noise on reported values")
    args = parser.parse_args()

    scenario = make_scenario(args.seed, args.flights, args.anomaly_rate,
                             depart_window_s=args.depart_window, noise=not args.no_noise)
    counts = write_dataset(args.out_dir, scenario)

    print(f"{len(scenario['flights'])} flights, {len(scenario['anomalies'])} with an anomaly"
          f" -> {args.out_dir}")
    for label, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {label:<24} {count:>9} states")