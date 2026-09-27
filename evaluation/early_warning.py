"""How early does Syren warn before an aircraft squawks an emergency?

Runs saved history days through the same pipeline a replay uses (trace parser, detection
engine, warning engine), one aircraft at a time on every CPU core, colours each report the
way the map does (red > yellow > orange > green), and measures how long before an aircraft
turned red it was already yellow or orange. A sample of normal aircraft goes through the same
engines, so the report also shows how often aircraft that never have an emergency get flagged.

    .venv/bin/python -m evaluation.early_warning                  # every day on disk
    .venv/bin/python -m evaluation.early_warning --dates 2024-07-20
    .venv/bin/python -m evaluation.early_warning --more 3         # plus 3 earlier days (downloads)

Writes data/evaluation/report.html, one image per emergency in data/evaluation/events/, and
caches each day's numbers in data/evaluation/days/<date>.json (--refresh recomputes them).
"""

import argparse
import concurrent.futures
import datetime
import functools
import glob
import gzip
import json
import math
import os
import statistics
import time
import zlib

from data.history import ARCHIVE_DIR, day_dir, days_on_disk, swap_day
from data.trace import CENTER, RADIUS_NM, _distance_nm, parse_trace
from detection.engine import build_default_engine
from ml.aircraft_warning import AircraftWarningEngine
from shared.flight_state import validate_flight_state

EVALUATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "evaluation")
EMERGENCY_SQUAWKS = {"7500", "7600", "7700"}
LOOKBACK_S = 20 * 60        # a warning counts if it came at most this long before red...
MAX_GAP_S = 5 * 60          # ...with no reception gap longer than this in between
MIN_RED_REPORTS = 2         # red must last this many reports, so a mistyped squawk doesn't count
FIGURE_BEFORE_S = 30 * 60   # event images show this long before red...
FIGURE_AFTER_S = 10 * 60    # ...and this long after
SAMPLE_PER_MILLE = 20       # normal aircraft (per 1000) run for the false-alarm numbers
CHANCE_EVERY_S = 5 * 60     # how often a normal flight is checked for the chance baseline
TAKEOFF_S = 5 * 60          # ignore warnings this long after takeoff...
LANDING_S = 5 * 60          # ...and this long before landing,
KEEP_NEAR_GROUND = {"AGGRESSIVE_NEAR_GROUND_SPEED"}  # except overspeed, which matters most there
LOW_ALTITUDE_FT = 3000      # a stretch starting/ending below this is a takeoff/landing even with no ground report
MIN_WARNING_S = 15          # orange that clears sooner than this is a flicker, not a warning
PROGRESS_EVERY = 2000       # print progress after this many aircraft files
DOWNLOAD_TRIES = 3          # attempts per day for --more downloads...
RETRY_WAIT_S = 30           # ...this far apart
NEW_LEG = 2                 # trace flag bit: a new flight starts at this point

GREEN, YELLOW, ORANGE, RED = "green", "yellow", "orange", "red"
WARNING_COLOURS = (YELLOW, ORANGE)


# Why a report is red, the same rules as alertKind in frontend/src/App.tsx; None if it isn't.
def red_reason(state, anomaly_types):
    status = state["status"]
    if status["squawk"] in EMERGENCY_SQUAWKS:
        return f"squawk {status['squawk']}"
    if status["emergency"] not in (None, "none"):
        return f"emergency {status['emergency']}"
    if "EMERGENCY_SQUAWK" in anomaly_types or state.get("anomaly") == "squawk":
        return "detector squawk"
    return None


# Run one aircraft's states (in time order) through fresh engines, like a replay does, and
# return one point per report: time, place, altitude, map colour, and what caused the colour.
def colour_timeline(states):
    detection, warnings = build_default_engine(), AircraftWarningEngine()
    timeline = []
    for state in states:
        try:
            result = detection.update_mapping(state)
            prediction = warnings.update(state)
        except (ValueError, TypeError):
            continue  # a report the engines refuse, e.g. out of order
        anomaly_types = [str(anomaly["type"]) for anomaly in result["anomalies"]]
        reason = red_reason(state, anomaly_types)
        if reason:
            colour, causes = RED, [reason]
        elif prediction.alert:
            colour, causes = YELLOW, [f"prediction {signal.type}" for signal in prediction.signals]
        elif anomaly_types or state.get("anomaly") not in (None, "none"):
            colour, causes = ORANGE, anomaly_types or [f"anomaly {state['anomaly']}"]
        else:
            colour, causes = GREEN, []
        position = state["position"]
        timeline.append({
            "t": state["timestamp"],
            "flight_id": state["flight_id"],
            "lat": position["latitude"],
            "lon": position["longitude"],
            "alt": position["altitude_baro_ft"] if position["altitude_baro_ft"] is not None
            else position["altitude_geom_ft"],
            "ground": position["on_ground"],
            "colour": colour,
            "causes": causes,
        })
    return timeline


# (start, end) index ranges of airborne stretches, broken by on-ground reports and by gaps over MAX_GAP_S.
def _stretches(timeline):
    ranges, start = [], None
    for i, point in enumerate(timeline):
        if point["ground"] or (start is not None and point["t"] - timeline[i - 1]["t"] > MAX_GAP_S):
            if start is not None:
                ranges.append((start, i))
            start = None
        if not point["ground"] and start is None:
            start = i
    if start is not None:
        ranges.append((start, len(timeline)))
    return ranges


# The point turned back to green, or kept orange with only the causes in `keep`.
def _muted(point, keep=frozenset()):
    kept = [cause for cause in point["causes"] if cause in keep]
    if point["colour"] == ORANGE and kept:
        return {**point, "causes": kept}
    return {**point, "colour": GREEN, "causes": []}


# Ignore warnings in the first takeoff_s after takeoff and the last landing_s before landing,
# except KEEP_NEAR_GROUND ones. A takeoff is a stretch that starts right after an on-ground
# report or below LOW_ALTITUDE_FT; landing likewise at the end. Red is left alone.
# (Ignoring the landing phase needs hindsight: the app can't know live that a landing is coming.)
def mute_takeoff_and_landing(timeline, takeoff_s=TAKEOFF_S, landing_s=LANDING_S):
    if not (takeoff_s or landing_s):
        return timeline
    timeline = list(timeline)
    for start, end in _stretches(timeline):
        first, last = timeline[start], timeline[end - 1]
        ground_before = start > 0 and timeline[start - 1]["ground"] and first["t"] - timeline[start - 1]["t"] <= MAX_GAP_S
        ground_after = end < len(timeline) and timeline[end]["ground"] and timeline[end]["t"] - last["t"] <= MAX_GAP_S
        took_off = ground_before or (first["alt"] is not None and first["alt"] < LOW_ALTITUDE_FT)
        landed = ground_after or (last["alt"] is not None and last["alt"] < LOW_ALTITUDE_FT)
        for i in range(start, end):
            point = timeline[i]
            if point["colour"] in WARNING_COLOURS and (
                    (took_off and point["t"] - first["t"] < takeoff_s)
                    or (landed and last["t"] - point["t"] < landing_s)):
                timeline[i] = _muted(point, KEEP_NEAR_GROUND)
    return timeline


# Ignore orange that clears in under min_s, measured from its first report to the next report
# that isn't orange. Yellow is left alone: a prediction alert is one report by design (followed
# by a 10-minute cooldown), so a length filter would remove every prediction.
def mute_brief(timeline, min_s=MIN_WARNING_S):
    if not min_s:
        return timeline
    timeline = list(timeline)
    i = 0
    while i < len(timeline):
        if timeline[i]["colour"] != ORANGE:
            i += 1
            continue
        j = i
        while (j + 1 < len(timeline) and timeline[j + 1]["colour"] == ORANGE
               and timeline[j + 1]["t"] - timeline[j]["t"] <= MAX_GAP_S):
            j += 1
        cleared = timeline[j + 1]["t"] if j + 1 < len(timeline) else timeline[j]["t"]
        if cleared - timeline[i]["t"] < min_s:
            for k in range(i, j + 1):
                timeline[k] = _muted(timeline[k])
        i = j + 1
    return timeline


# The yellow/orange reports in the LOOKBACK_S before timeline[i], newest first, stopping at a
# reception gap longer than MAX_GAP_S. Emergencies and chance moments use the same window.
def warnings_before(timeline, i):
    found = []
    j = i - 1
    while (j >= 0 and timeline[j + 1]["t"] - timeline[j]["t"] <= MAX_GAP_S
           and timeline[i]["t"] - timeline[j]["t"] <= LOOKBACK_S):
        if timeline[j]["colour"] in WARNING_COLOURS:
            found.append(timeline[j])
        j -= 1
    return found


# The first time the aircraft turns red while airborne and stays red for MIN_RED_REPORTS
# reports, with the earliest yellow/orange report in the window before it and every kind of
# warning seen in that window. None if it never does.
def find_event(timeline):
    for i, point in enumerate(timeline):
        if point["colour"] != RED or point["ground"]:
            continue
        run = timeline[i:i + MIN_RED_REPORTS]
        if len(run) < MIN_RED_REPORTS or any(p["colour"] != RED for p in run):
            continue
        before = warnings_before(timeline, i)
        warning = before[-1] if before else None  # the earliest
        return {
            "red_index": i,
            "red_time": point["t"],
            "reason": point["causes"][0],
            "warning": warning and {"t": warning["t"], "colour": warning["colour"],
                                    "causes": warning["causes"]},
            "lead_s": round(point["t"] - warning["t"], 1) if warning else None,
            "window_causes": sorted({cause for p in before for cause in p["causes"]}),
        }
    return None


# The chance baseline for an aircraft with no emergency: every `every_s` of airborne flight,
# would the same window have held a warning? {"moments", "warned", "by_cause": {cause: moments}}
def chance_moments(timeline, every_s=CHANCE_EVERY_S):
    moments, warned, by_cause = 0, 0, {}
    next_t = None
    for i, point in enumerate(timeline):
        if point["ground"] or (next_t is not None and point["t"] < next_t):
            continue
        next_t = point["t"] + every_s
        moments += 1
        before = warnings_before(timeline, i)
        if before:
            warned += 1
        for cause in {cause for p in before for cause in p["causes"]}:
            by_cause[cause] = by_cause.get(cause, 0) + 1
    return {"moments": moments, "warned": warned, "by_cause": by_cause}


# How much an aircraft was flagged: airborne time, and how many times it went from green to
# yellow/orange (a "warning onset"), by what caused it.
def alert_burden(timeline):
    airborne_s, onsets, by_cause = 0.0, 0, {}
    previous = None
    for point in timeline:
        if previous is not None and not point["ground"] and point["t"] - previous["t"] <= MAX_GAP_S:
            airborne_s += point["t"] - previous["t"]
        started = point["colour"] in WARNING_COLOURS and (
            previous is None or previous["colour"] not in WARNING_COLOURS)
        if started:
            onsets += 1
            cause = point["causes"][0] if point["causes"] else point["colour"]
            by_cause[cause] = by_cause.get(cause, 0) + 1
        previous = point
    return {"airborne_s": round(airborne_s, 1), "onsets": onsets, "by_cause": by_cause}


# Cheap first look at one raw trace, before building any states: is the aircraft ever inside
# the area, and does it ever report an emergency (squawk or status) there?
def _scan_trace(trace, in_area):
    seen, emergency, details = False, False, {}
    for point in trace["trace"]:
        if point[6] & NEW_LEG:
            details = {}
        if point[8]:
            details = {**details, **point[8]}
        if not in_area(point[1], point[2]):
            continue
        seen = True
        if details.get("squawk") in EMERGENCY_SQUAWKS or details.get("emergency") not in (None, "none"):
            emergency = True
            break
    return seen, emergency


# One aircraft's day, run in a worker process. Aircraft that report an emergency, and a
# fixed sample of the others, go through the engines; the rest are only counted.
def _evaluate_trace(path, area, sample_per_mille, filters=(TAKEOFF_S, LANDING_S, MIN_WARNING_S)):
    with gzip.open(path) as f:
        trace = json.load(f)
    if area is None:
        in_area = lambda lat, lon: True
    else:
        lat0, lon0, radius = area
        in_area = lambda lat, lon: _distance_nm(lat0, lon0, lat, lon) <= radius
    seen, emergency = _scan_trace(trace, in_area)
    if not seen:
        return None
    icao24 = trace["icao"]
    sampled = zlib.crc32(icao24.encode()) % 1000 < sample_per_mille
    if not (emergency or sampled):
        return {"icao24": icao24}

    states = [state for state in parse_trace(path, keep=lambda t, lat, lon: in_area(lat, lon))
              if not validate_flight_state(state)]
    takeoff_s, landing_s, min_warning_s = filters
    timeline = mute_brief(mute_takeoff_and_landing(colour_timeline(states), takeoff_s, landing_s), min_warning_s)
    result = {
        "icao24": icao24,
        "flight_id": states[-1]["flight_id"] if states else icao24,
        "type_code": trace.get("t"),
        "registration": trace.get("r"),
        "burden": alert_burden(timeline),
        "chance": chance_moments(timeline),
        "sampled": sampled and not emergency,
        "emergency": emergency,
    }
    if emergency:
        event = find_event(timeline)
        if event:
            result["flight_id"] = timeline[event.pop("red_index")]["flight_id"]  # callsign when it went red
            red_time = event["red_time"]
            event["timeline"] = [p for p in timeline
                                 if red_time - FIGURE_BEFORE_S <= p["t"] <= red_time + FIGURE_AFTER_S]
        result["event"] = event
    return result


# Evaluate one day on disk: every aircraft's trace, spread over `workers` processes.
def evaluate_day(date, root=ARCHIVE_DIR, area=(CENTER[0], CENTER[1], RADIUS_NM),
                 sample_per_mille=SAMPLE_PER_MILLE, workers=None,
                 filters=(TAKEOFF_S, LANDING_S, MIN_WARNING_S)):
    folder = day_dir(date, root)
    if folder is None:
        raise LookupError(f"{date} is not on disk")
    files = glob.glob(os.path.join(folder, "traces", "*", "trace_full_*.json"))
    check = functools.partial(_evaluate_trace, area=area, sample_per_mille=sample_per_mille, filters=filters)
    in_area, events, excluded, sample, chance = 0, [], {"lifeguard": 0, "brief_or_on_ground": 0}, [], []
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        for done, result in enumerate(pool.map(check, files, chunksize=200), 1):
            if done % PROGRESS_EVERY == 0 or done == len(files):
                print(f"\r  checked {done:,} / {len(files):,} aircraft ({100 * done // len(files)}%), "
                      f"{len(events)} emergencies so far", end="" if done < len(files) else "\n", flush=True)
            if result is None:
                continue
            in_area += 1
            if result.get("sampled"):
                sample.append(result["burden"])
                chance.append(result["chance"])
            if not result.get("emergency"):
                continue
            event = result.pop("event")
            if event is None:
                excluded["brief_or_on_ground"] += 1
            elif event["reason"] == "emergency lifeguard":
                excluded["lifeguard"] += 1
            else:
                events.append({**{k: result[k] for k in ("icao24", "flight_id", "type_code", "registration")},
                               "date": date, **event})
    return {
        "date": date,
        "area": "worldwide" if area is None else f"{area[2]:g} nm around {area[0]}, {area[1]}",
        "filters": {"takeoff_s": filters[0], "landing_s": filters[1], "min_warning_s": filters[2]},
        "aircraft": in_area,
        "events": events,
        "excluded": excluded,
        "sample": {
            "aircraft": len(sample),
            "flagged": sum(1 for burden in sample if burden["onsets"] > 0),
            "airborne_hours": round(sum(burden["airborne_s"] for burden in sample) / 3600, 2),
            "onsets": sum(burden["onsets"] for burden in sample),
            "by_cause": _add_counts(burden["by_cause"] for burden in sample),
        },
        "chance": {
            "moments": sum(c["moments"] for c in chance),
            "warned": sum(c["warned"] for c in chance),
            "by_cause": _add_counts(c["by_cause"] for c in chance),
        },
    }


# 95% Wilson score interval for count/total, as percentages: how sure a rate is on few events.
def wilson_interval(count, total, z=1.96):
    if not total:
        return None
    rate = count / total
    centre = (rate + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def _add_counts(counters):
    total = {}
    for counter in counters:
        for key, value in counter.items():
            total[key] = total.get(key, 0) + value
    return dict(sorted(total.items(), key=lambda item: -item[1]))


# The headline numbers over all evaluated days.
def summarise(days):
    events = [event for day in days for event in day["events"]]
    leads = sorted(event["lead_s"] for event in events if event["lead_s"] is not None)
    sample_aircraft = sum(day["sample"]["aircraft"] for day in days)
    airborne_hours = sum(day["sample"]["airborne_hours"] for day in days)
    onsets = sum(day["sample"]["onsets"] for day in days)

    def share(count, total):
        return round(100 * count / total, 1) if total else None

    # Each kind of warning: how often it came in the window before an emergency, how often the
    # same window held it at an ordinary moment of a normal flight, and the ratio ("lift").
    # A lift near 1 means the warning is no more common before emergencies than any other time.
    moments = sum(day["chance"]["moments"] for day in days)
    chance_warned = sum(day["chance"]["warned"] for day in days)
    before_counts = _add_counts({cause: 1 for cause in event["window_causes"]} for event in events)
    chance_counts = _add_counts(day["chance"]["by_cause"] for day in days)
    comparison = []
    for cause in set(before_counts) | set(chance_counts):
        before = before_counts.get(cause, 0) / len(events) if events else 0.0
        normal = chance_counts.get(cause, 0) / moments if moments else 0.0
        comparison.append({
            "cause": cause,
            "before_emergencies_pct": round(100 * before, 1),
            "normal_moments_pct": round(100 * normal, 1),
            "lift": round(before / normal, 1) if normal else None,
        })
    comparison.sort(key=lambda row: (row["lift"] is None, -(row["lift"] or 0), -row["before_emergencies_pct"]))
    warned_rate = len(leads) / len(events) if events else None
    chance_rate = chance_warned / moments if moments else None

    return {
        "days": [day["date"] for day in days],
        "aircraft": sum(day["aircraft"] for day in days),
        "events": len(events),
        "excluded": _add_counts(day["excluded"] for day in days),
        "warned": len(leads),
        "warned_pct": share(len(leads), len(events)),
        "warned_pct_95": wilson_interval(len(leads), len(events)),
        "warned_1min_pct": share(sum(1 for lead in leads if lead >= 60), len(events)),
        "warned_5min_pct": share(sum(1 for lead in leads if lead >= 300), len(events)),
        "lead_median_s": statistics.median(leads) if leads else None,
        "lead_mean_s": round(statistics.fmean(leads), 1) if leads else None,
        "lead_quartiles_s": statistics.quantiles(leads, n=4) if len(leads) >= 2 else None,
        "lead_max_s": leads[-1] if leads else None,
        "first_warning_by": _add_counts(
            {f"{event['warning']['colour']}: {event['warning']['causes'][0] if event['warning']['causes'] else '?'}": 1}
            for event in events if event["warning"]),
        "sample_aircraft": sample_aircraft,
        "sample_flagged_pct": share(sum(day["sample"]["flagged"] for day in days), sample_aircraft),
        "sample_onsets_per_hour": round(onsets / airborne_hours, 2) if airborne_hours else None,
        "sample_by_cause": _add_counts(day["sample"]["by_cause"] for day in days),
        "chance_moments": moments,
        "chance_warned_pct": share(chance_warned, moments),
        "lift": round(warned_rate / chance_rate, 2) if warned_rate is not None and chance_rate else None,
        "by_warning": comparison,
    }


# The days to start from: the given ones, or every day on disk plus every day already evaluated
# with these settings (so a day whose files were swapped out still counts).
def saved_dates(dates, root, cache_dir=None, settings=None):
    if not dates:
        on_disk = days_on_disk(root)
        dates = on_disk["fixed"] + ([on_disk["swap"]] if on_disk["swap"] else [])
        if cache_dir and settings:
            suffix = f"_{settings}.json"
            dates += [name[:-len(suffix)] for name in os.listdir(cache_dir) if name.endswith(suffix)]
    return sorted(set(dates))


# Download a day into the swap slot, retrying network hiccups. False if it can't be had.
def fetch_day(date):
    for attempt in range(1, DOWNLOAD_TRIES + 1):
        try:
            swap_day(date)
            return True
        except (LookupError, ValueError) as error:  # no release for that day, or not an archive day
            print(f"{date}: skipped ({error})")
            return False
        except OSError as error:  # network trouble; urllib's URLError is an OSError
            print(f"{date}: download failed ({error}), try {attempt} of {DOWNLOAD_TRIES}")
            if attempt < DOWNLOAD_TRIES:
                time.sleep(RETRY_WAIT_S)
    return False


def main():
    from evaluation.figures import write_report

    parser = argparse.ArgumentParser(description="Measure how early Syren warns before emergencies.")
    parser.add_argument("--dates", nargs="*", default=[], help="YYYY-MM-DD days (default: every day on disk)")
    parser.add_argument("--more", type=int, default=0,
                        help="also evaluate this many earlier days, downloading each into the swap "
                             "slot (about 10 minutes and 3 GB each; replaces the current swap day)")
    parser.add_argument("--worldwide", action="store_true", help="every aircraft, not just the app's US area")
    parser.add_argument("--workers", type=int, default=None, help="worker processes (default: every CPU core)")
    parser.add_argument("--sample", type=int, default=SAMPLE_PER_MILLE,
                        help="normal aircraft per 1000 used for the false-alarm numbers")
    parser.add_argument("--takeoff", type=float, default=TAKEOFF_S / 60,
                        help="ignore warnings this many minutes after takeoff, except near-ground overspeed (0: off)")
    parser.add_argument("--landing", type=float, default=LANDING_S / 60,
                        help="ignore warnings this many minutes before landing, except near-ground overspeed (0: off)")
    parser.add_argument("--min-warning", type=float, default=MIN_WARNING_S,
                        help="ignore orange that clears in fewer seconds than this (0: off)")
    parser.add_argument("--refresh", action="store_true", help="recompute days that are already cached")
    parser.add_argument("--out", default=EVALUATION_DIR)
    args = parser.parse_args()

    area = None if args.worldwide else (CENTER[0], CENTER[1], RADIUS_NM)
    filters = (args.takeoff * 60, args.landing * 60, args.min_warning)
    settings = (f"{'world' if area is None else 'us'}_s{args.sample}"
                f"_to{filters[0]:g}_ld{filters[1]:g}_min{filters[2]:g}")
    os.makedirs(os.path.join(args.out, "days"), exist_ok=True)
    # One day's results: from the cache, or downloaded if needed and evaluated. None if unavailable.
    def load_or_evaluate(date):
        cache = os.path.join(args.out, "days", f"{date}_{settings}.json")  # one cache per setting
        if os.path.exists(cache) and not args.refresh:
            with open(cache, encoding="utf-8") as f:
                print(f"{date}: cached")
                return json.load(f)
        if day_dir(date) is None and not fetch_day(date):
            return None
        started = datetime.datetime.now()
        day = evaluate_day(date, area=area, sample_per_mille=args.sample, workers=args.workers, filters=filters)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(day, f)
        warned = sum(1 for event in day["events"] if event["lead_s"] is not None)
        print(f"{date}: {day['aircraft']} aircraft, {len(day['events'])} emergencies, "
              f"{warned} warned early ({(datetime.datetime.now() - started).seconds} s)")
        return day

    chosen = saved_dates(args.dates, ARCHIVE_DIR, os.path.join(args.out, "days"), settings)
    target = len(chosen) + args.more
    print(f"evaluating {target} days: {', '.join(chosen)}"
          + (f" and {args.more} earlier" if args.more else ""))
    days = []
    for date in chosen:
        print(f"[{len(days) + 1}/{target}] {date}")
        day = load_or_evaluate(date)
        if day:
            days.append(day)
    # --more: step back a day at a time until that many extra days are done (giving up after 3x the tries)
    date, extra, tries = datetime.date.fromisoformat(chosen[0]), 0, 0
    while extra < args.more and tries < 3 * args.more:
        date -= datetime.timedelta(days=1)
        tries += 1
        if date.isoformat() in chosen:
            continue
        print(f"[{len(days) + 1}/{target}] {date.isoformat()}")
        day = load_or_evaluate(date.isoformat())
        if day:
            days.append(day)
            extra += 1

    if not days:
        raise SystemExit("no days evaluated")
    summary = write_report(days, summarise(days), args.out)
    print(f"\n{summary['warned']} of {summary['events']} emergencies warned early "
          f"({summary['warned_pct']}%), median lead {summary['lead_median_s']} s")
    print(f"chance level: {summary['chance_warned_pct']}% of ordinary moments in normal flights had a "
          f"warning in the same window (lift {summary['lift']})")
    print(f"report: {os.path.join(args.out, 'report.html')}")


if __name__ == "__main__":
    main()
