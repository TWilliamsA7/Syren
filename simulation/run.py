import copy
import json
import sys

from shared.flight_state import validate_flight_state
from simulation.anomalies import apply_anomalies, check_anomalies, ground_truth
from simulation.flight_plan import AIRPORTS, FlightPlan

STALE_AFTER_S = 20       # same rule as data/live_adapter.py
SIGNAL_TIMEOUT_S = 60    # live feeds drop an aircraft about a minute after its last message


def load_scenario(path):
    with open(path) as f:
        return json.load(f)


def _depart(flight):
    plan = FlightPlan(
        origin=AIRPORTS[flight["origin"]],
        destination=AIRPORTS[flight["destination"]],
        cruise_altitude_ft=flight["cruise_altitude_ft"],
        cruise_speed_kts=flight.get("cruise_speed_kts", 450),
        squawk=flight.get("squawk", "1200"),
    )
    ac = plan.spawn(
        flight["icao24"], flight["callsign"], flight["type_code"],
        flight.get("category", "A3"),
    )
    return plan, ac


def _aged(state, age_s):
    """The last state heard from an aircraft, as a live feed shows it age_s later."""
    aged = copy.deepcopy(state)
    aged["status"]["seen_age_s"] = age_s
    aged["position"]["stale"] = age_s > STALE_AFTER_S
    return aged


def run_scenario(scenario):
    """Yield FlightStates tick by tick; a silent aircraft re-reports its last position for a minute."""
    dt = scenario.get("dt", 1.0)
    start_time = scenario["start_time"]
    anomalies = scenario.get("anomalies", [])
    check_anomalies(anomalies)
    waiting = sorted(scenario["flights"], key=lambda f: f["depart_at_s"])
    active = []
    last_heard = {}

    for tick in range(int(scenario["duration_s"] / dt) + 1):
        t = tick * dt
        while waiting and waiting[0]["depart_at_s"] <= t:
            active.append(_depart(waiting.pop(0)))

        for plan, ac in active:
            plan.update(ac)
        apply_anomalies(anomalies, t, {ac.icao24: ac for _, ac in active})

        for plan, ac in active:
            if ac.transponder_on:
                state = ac.to_flight_state(start_time + t)
                last_heard[ac.icao24] = state
                yield state
            elif ac.icao24 in last_heard:
                age_s = start_time + t - last_heard[ac.icao24]["timestamp"]
                if age_s <= SIGNAL_TIMEOUT_S:
                    yield _aged(last_heard[ac.icao24], age_s)

        active = [(plan, ac) for plan, ac in active
                  if not (plan.finished and ac.speed_kts == 0)]
        for plan, ac in active:
            ac.step(dt)


def summary_line(state, start_time):
    alt = state["position"]["altitude_baro_ft"]
    return (f"t={state['timestamp'] - start_time:>5.0f} {state['flight_id']:<7}"
            f" ALT={'GND' if alt is None else alt:>5}"
            f" VS={state['kinematics']['vertical_rate_baro_fpm']:>5}"
            f" SPD={state['kinematics']['ground_speed_kts']:>5.0f}"
            f" HDG={state['kinematics']['track_deg']:>6}")


if __name__ == "__main__":
    scenario = load_scenario(sys.argv[1])
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    truth_path = sys.argv[3] if len(sys.argv) > 3 else None
    start_time = scenario["start_time"]

    if truth_path:
        with open(truth_path, "w") as f:
            json.dump(ground_truth(scenario), f, indent=2)

    out = open(out_path, "w") if out_path else None
    count = 0
    for state in run_scenario(scenario):
        count += 1
        errors = validate_flight_state(state)
        if errors:
            print(state["icao24"], errors)
        if out:
            out.write(json.dumps(state) + "\n")
        elif (state["timestamp"] - start_time) % 60 == 0:
            print(summary_line(state, start_time))
    if out:
        out.close()
    print(f"{count} FlightStates")