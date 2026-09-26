from shared.flight_state import validate_flight_state
from simulation.flight_plan import AIRPORTS, FlightPlan, distance_and_bearing


def fly(plan, ac, max_seconds=3 * 3600):
    """Fly until landed; return the phases seen and the highest altitude."""
    phases = [plan.phase]
    highest = ac.altitude_ft
    for t in range(max_seconds):
        plan.update(ac)
        if plan.phase != phases[-1]:
            phases.append(plan.phase)
        if plan.finished:
            break
        ac.step(1.0)
        highest = max(highest, ac.altitude_ft)
        assert validate_flight_state(ac.to_flight_state(t)) == [], t
    return phases, highest


def distance_to(ac, code):
    airport = AIRPORTS[code]
    distance, _ = distance_and_bearing(
        ac.latitude, ac.longitude, airport.latitude, airport.longitude)
    return distance


def test_distance_and_bearing():
    distance, bearing = distance_and_bearing(28.4294, -81.3090, 27.9755, -82.5332)
    assert 68 < distance < 73, distance
    assert 244 < bearing < 250, bearing
    _, north = distance_and_bearing(28.0, -81.0, 29.0, -81.0)
    assert north == 0, north
    _, east = distance_and_bearing(28.0, -81.0, 28.0, -80.0)
    assert abs(east - 90) < 1e-9, east


def test_long_flight_all_phases():
    plan = FlightPlan(AIRPORTS["MIA"], AIRPORTS["JAX"], cruise_altitude_ft=37000)
    ac = plan.spawn("~5f0001", "SYN101", "A320")
    phases, highest = fly(plan, ac)
    assert phases == ["takeoff", "climb", "cruise", "descent", "approach", "landed"], phases
    assert abs(highest - 37000) < 1, highest
    assert distance_to(ac, "JAX") < 1, distance_to(ac, "JAX")
    assert ac.on_ground


def test_short_hop_skips_cruise():
    plan = FlightPlan(AIRPORTS["MCO"], AIRPORTS["TPA"], cruise_altitude_ft=35000)
    ac = plan.spawn("~5f0002", "SYN202", "B738")
    phases, highest = fly(plan, ac)
    assert "cruise" not in phases, phases
    assert phases[-1] == "landed", phases
    assert highest < 35000, highest
    assert distance_to(ac, "TPA") < 1, distance_to(ac, "TPA")


if __name__ == "__main__":
    test_distance_and_bearing()
    test_long_flight_all_phases()
    test_short_hop_skips_cruise()
    print("all tests passed")