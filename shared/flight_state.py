import re
ORIGINS = {"live", "history", "sim"}
EMERGENCIES = {"none", "general", "lifeguard", "minfuel", "nordo", "unlawful", "downed"}
SOURCES = {"adsb_icao", "adsr_icao", "mlat"}
TOP_KEYS = (
    "timestamp", "icao24", "flight_id",
    "aircraft", "position", "kinematics", "nav", "status",
    "origin",
)
SECTION_KEYS = {
    "aircraft": ("registration", "type_code", "category"),
    "position": (
        "latitude", "longitude", "altitude_baro_ft", "altitude_geom_ft",
        "on_ground", "source", "accuracy_m", "stale",
    ),
    "kinematics": (
        "ground_speed_kts", "track_deg",
        "vertical_rate_baro_fpm", "vertical_rate_geom_fpm",
    ),
    "nav": ("selected_altitude_ft",),
    "status": ("squawk", "emergency", "seen_age_s"),
}
SQUAWK_RE = re.compile(r"[0-7]{4}")
ICAO24_RE = re.compile(r"~?[0-9a-f]{6}")
CATEGORY_RE = re.compile(r"[A-Z][0-9]")

FIELD_RULES = (
    # (section, key, kind, required, check)
    (None, "timestamp", "number", True, None),
    (None, "icao24", "str", True, ICAO24_RE.fullmatch),
    (None, "flight_id", "str", True, None),
    ("aircraft", "registration", "str", False, None),
    ("aircraft", "type_code", "str", False, None),
    ("aircraft", "category", "str", False, CATEGORY_RE.fullmatch),
    ("position", "latitude", "number", True, lambda v: -90 <= v <= 90),
    ("position", "longitude", "number", True, lambda v: -180 <= v <= 180),
    ("position", "altitude_baro_ft", "number", False, None),
    ("position", "altitude_geom_ft", "number", False, None),
    ("position", "on_ground", "bool", True, None),
    ("position", "source", "str", True, lambda v: v in SOURCES),
    ("position", "accuracy_m", "number", False, lambda v: v >= 0),
    ("position", "stale", "bool", True, None),
    ("kinematics", "ground_speed_kts", "number", False, lambda v: v >= 0),
    ("kinematics", "track_deg", "number", False, lambda v: 0 <= v < 360),
    ("kinematics", "vertical_rate_baro_fpm", "number", False, None),
    ("kinematics", "vertical_rate_geom_fpm", "number", False, None),
    ("nav", "selected_altitude_ft", "number", False, None),
    ("status", "squawk", "str", False, SQUAWK_RE.fullmatch),
    ("status", "emergency", "str", False, lambda v: v in EMERGENCIES),
    ("status", "seen_age_s", "number", False, lambda v: v >= 0),
    (None, "origin", "str", True, lambda v: v in ORIGINS),
)

def round_altitude(ft):
    if ft is None:
        return None
    return int(round(ft/25)*25)

def round_vertical_rate(fpm):
    if fpm is None:
        return None
    return int(round(fpm/64)*64)

def round_track(deg):
    if deg is None:
        return None
    deg = round(deg%360, 2)
    if deg == 360.0:
        return 0.0
    return deg

def make_flight_state(
    *,
    timestamp, icao24, latitude, longitude, on_ground, source, origin,
    flight_id=None,
    registration=None, type_code=None, category=None,
    altitude_baro_ft=None, altitude_geom_ft=None, accuracy_m=None, stale=False,
    ground_speed_kts=None, track_deg=None,
    vertical_rate_baro_fpm=None, vertical_rate_geom_fpm=None,
    selected_altitude_ft=None,
    squawk=None, emergency=None, seen_age_s=None,):
    if flight_id is not None:
        flight_id = flight_id.strip()
    if not flight_id:
        flight_id = icao24

    return {
        "timestamp": timestamp,
        "icao24": icao24,
        "flight_id": flight_id,
        "aircraft": {
            "registration": registration,
            "type_code": type_code,
            "category": category,
        },
        "position": {
            "latitude": latitude,
            "longitude": longitude,
            "altitude_baro_ft": altitude_baro_ft,
            "altitude_geom_ft": altitude_geom_ft,
            "on_ground": on_ground,
            "source": source,
            "accuracy_m": accuracy_m,
            "stale": stale,
        },
        "kinematics": {
            "ground_speed_kts": ground_speed_kts,
            "track_deg": track_deg,
            "vertical_rate_baro_fpm": vertical_rate_baro_fpm,
            "vertical_rate_geom_fpm": vertical_rate_geom_fpm,
        },
        "nav": {
            "selected_altitude_ft": selected_altitude_ft,
        },
        "status": {
            "squawk": squawk,                  
            "emergency": emergency,               
            "seen_age_s": seen_age_s               
        },
        "origin" : origin,
    }

def _is_kind(value, kind):
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "bool":
        return isinstance(value, bool)
    return isinstance(value, str)

def validate_flight_state(state) :
    errors = []
    if not isinstance(state, dict):
        return ["state is not a dict"]

    missing = set(TOP_KEYS) - state.keys()
    extra = state.keys()-set(TOP_KEYS)
    for key in sorted(missing):
        errors.append(f"missing key: {key}")
    for key in sorted(extra):
        errors.append(f"unexpected key: {key}")
    for name, keys in SECTION_KEYS.items():
        if name not in state:
            continue
        if not isinstance(state[name], dict):
            errors.append(f"{name} is not a dict")
            continue
        for key in sorted(set(keys) - state[name].keys()):
            errors.append(f"missing key: {name}.{key}")
        for key in sorted(state[name].keys() - set(keys)):
            errors.append(f"unexpected key: {name}.{key}")
    if errors:
        return errors

    for section, key, kind, required, check in FIELD_RULES:
        value = state[key] if section is None else state[section][key]
        path = key if section is None else f"{section}.{key}"
        if value is None:
            if required:
                errors.append(f"{path} is required")
            continue
        if not _is_kind(value, kind):
            errors.append(f"{path} should be {kind}, got {value!r}")
            continue
        if check is not None and not check(value):
            errors.append(f"{path} has invalid value {value!r}")
    
    return errors
     