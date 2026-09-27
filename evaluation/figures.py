"""Images and the HTML report for evaluation.early_warning, using only the standard library.

Each emergency gets one SVG image: its track on the left and its altitude on the right, both
coloured the way the map colours the aircraft, from 30 minutes before it turned red to 10 after.
"""

import datetime
import glob
import html
import json
import math
import os

from evaluation.early_warning import FIGURE_AFTER_S, FIGURE_BEFORE_S, LOOKBACK_S, MAX_GAP_S, MIN_RED_REPORTS

COLOURS = {"green": "#36F6B4", "yellow": "#FACC15", "orange": "#FB923C", "red": "#F43F5E"}
BACKGROUND, PANEL, GRID, TEXT, MUTED = "#0F172A", "#111827", "#334155", "#F8FAFC", "#94A3B8"
FONT = "font-family=\"IBM Plex Sans, system-ui, sans-serif\""
LEGEND = (("green", "normal"), ("yellow", "prediction warning"), ("orange", "detector warning"),
          ("red", "emergency squawk"))


def _esc(text):
    return html.escape(str(text))


# "45 s", "6 min 30 s"
def format_duration(seconds):
    if seconds is None:
        return "—"
    seconds = round(seconds)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes} min {seconds} s" if minutes else f"{seconds} s"


# Lines between consecutive points, each in the colour of the point it leads to (so the colour
# changes where the aircraft's colour changes), with no line across a reception gap.
def _segments(points, x, y):
    lines = []
    for a, b in zip(points, points[1:]):
        if b["t"] - a["t"] > MAX_GAP_S:
            continue
        lines.append(f'<line x1="{x(a):.1f}" y1="{y(a):.1f}" x2="{x(b):.1f}" y2="{y(b):.1f}" '
                     f'stroke="{COLOURS[b["colour"]]}" stroke-width="2.5" stroke-linecap="round"/>')
    return lines


# A ring on the track with its label above (dy < 0) or below (dy > 0), so two close markers don't collide.
def _marker(cx, cy, colour, label, dy):
    return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="none" stroke="{COLOURS[colour]}" stroke-width="2.5"/>'
            f'<text x="{cx:.1f}" y="{cy + dy:.1f}" fill="{COLOURS[colour]}" font-size="12" '
            f'text-anchor="middle" {FONT}>{_esc(label)}</text>')


# The track panel: latitude/longitude, north up, scaled to fit the box.
def _track_panel(points, event, left, top, width, height):
    parts = [f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="6" fill="{PANEL}"/>',
             f'<text x="{left + 12}" y="{top + 20}" fill="{MUTED}" font-size="12" {FONT}>Track</text>']
    points = [p for p in points if p["lat"] is not None and p["lon"] is not None]
    if len(points) < 2:
        return parts
    k = math.cos(math.radians(sum(p["lat"] for p in points) / len(points)))
    xs, ys = [p["lon"] * k for p in points], [p["lat"] for p in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.02)
    pad = 36
    scale = (min(width, height) - 2 * pad) / span
    x_offset = left + (width - (max(xs) - min(xs)) * scale) / 2
    y_offset = top + (height - (max(ys) - min(ys)) * scale) / 2
    x = lambda p: x_offset + (p["lon"] * k - min(xs)) * scale
    y = lambda p: y_offset + (max(ys) - p["lat"]) * scale
    parts += _segments(points, x, y)

    red = next((p for p in points if p["t"] == event["red_time"]), None)
    if event["warning"]:
        warned = next((p for p in points if p["t"] == event["warning"]["t"]), None)
        if warned:
            parts.append(_marker(x(warned), y(warned), warned["colour"], "first warning", dy=-12))
    if red:
        parts.append(_marker(x(red), y(red), "red", "emergency", dy=22))
    return parts


# The altitude panel: altitude against minutes relative to turning red.
def _altitude_panel(points, event, left, top, width, height):
    parts = [f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="6" fill="{PANEL}"/>',
             f'<text x="{left + 12}" y="{top + 20}" fill="{MUTED}" font-size="12" {FONT}>Altitude</text>']
    points = [{**p, "alt": 0 if p["alt"] is None and p["ground"] else p["alt"]} for p in points]
    points = [p for p in points if p["alt"] is not None]
    if len(points) < 2:
        return parts
    plot_left, plot_right, plot_top, plot_bottom = left + 64, left + width - 16, top + 34, top + height - 34
    highest = max(p["alt"] for p in points)
    step_ft = next(step for step in (1000, 2000, 5000, 10000) if highest <= step * 4)  # 4 grid steps at most
    top_ft = max(step_ft, math.ceil(highest / step_ft) * step_ft)
    start_min, end_min = -FIGURE_BEFORE_S / 60, FIGURE_AFTER_S / 60
    x_of_min = lambda m: plot_left + (m - start_min) / (end_min - start_min) * (plot_right - plot_left)
    x = lambda p: x_of_min((p["t"] - event["red_time"]) / 60)
    y = lambda p: plot_bottom - p["alt"] / top_ft * (plot_bottom - plot_top)

    for feet in range(0, top_ft + 1, step_ft):  # altitude grid
        gy = plot_bottom - feet / top_ft * (plot_bottom - plot_top)
        parts.append(f'<line x1="{plot_left}" y1="{gy:.1f}" x2="{plot_right}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{plot_left - 6}" y="{gy + 4:.1f}" fill="{MUTED}" font-size="11" text-anchor="end" {FONT}>{feet:,.0f}</text>')
    for minute in range(int(start_min), int(end_min) + 1, 5):  # time axis
        gx = x_of_min(minute)
        parts.append(f'<text x="{gx:.1f}" y="{plot_bottom + 18}" fill="{MUTED}" font-size="11" text-anchor="middle" {FONT}>'
                     f'{"+" if minute > 0 else ""}{minute}</text>')
    parts.append(f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{top + height - 2}" fill="{MUTED}" font-size="11" '
                 f'text-anchor="middle" {FONT}>minutes from emergency</text>')

    # A dashed line with its label beside it: the emergency's to the right, the warning's to the
    # left, so the two labels never overlap however close the lines are.
    def marker_line(minute, colour, label, side):
        gx = x_of_min(minute)
        anchor, dx = ("start", 4) if side == "right" else ("end", -4)
        return (f'<line x1="{gx:.1f}" y1="{plot_top}" x2="{gx:.1f}" y2="{plot_bottom}" stroke="{COLOURS[colour]}" '
                f'stroke-width="1.5" stroke-dasharray="4 4"/>'
                f'<text x="{gx + dx:.1f}" y="{plot_bottom - 8}" fill="{COLOURS[colour]}" font-size="11" '
                f'text-anchor="{anchor}" {FONT}>{_esc(label)}</text>')

    parts.append(marker_line(0, "red", "emergency", "right"))
    if event["warning"]:
        warned_min = (event["warning"]["t"] - event["red_time"]) / 60
        parts.append(marker_line(warned_min, event["warning"]["colour"], f"warning {warned_min:.1f} min", "left"))
    parts += _segments(points, x, y)
    return parts


# One emergency as an SVG image.
def event_svg(event):
    width, height = 960, 400
    lead = (f"warned {format_duration(event['lead_s'])} before"
            if event["lead_s"] is not None else "no early warning")
    cause = f" · first warning: {', '.join(event['warning']['causes'])}" if event["warning"] else ""
    title = f"{event['flight_id']} ({event['icao24']}) · {event['date']} · {event['reason']} · {lead}"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
             f'<text x="20" y="28" fill="{TEXT}" font-size="15" font-weight="600" {FONT}>{_esc(title)}</text>',
             f'<text x="20" y="48" fill="{MUTED}" font-size="12" {FONT}>'
             f'{_esc((event.get("type_code") or "unknown type") + cause)}</text>']
    parts += _track_panel(event["timeline"], event, 20, 62, 440, 300)
    parts += _altitude_panel(event["timeline"], event, 480, 62, 460, 300)
    for i, (colour, label) in enumerate(LEGEND):
        lx = 20 + i * 180
        parts.append(f'<rect x="{lx}" y="376" width="14" height="4" rx="2" fill="{COLOURS[colour]}"/>'
                     f'<text x="{lx + 20}" y="382" fill="{MUTED}" font-size="11" {FONT}>{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# Bar chart: how many emergencies were warned how far ahead, plus those with no warning.
def lead_histogram_svg(events):
    bins = [("< 1 min", 0, 60), ("1–2 min", 60, 120), ("2–5 min", 120, 300),
            ("5–10 min", 300, 600), ("10–20 min", 600, LOOKBACK_S + 1)]
    counts = [sum(1 for e in events if e["lead_s"] is not None and low <= e["lead_s"] < high)
              for _, low, high in bins]
    labels = [label for label, _, _ in bins] + ["no warning"]
    counts.append(sum(1 for e in events if e["lead_s"] is None))
    width, height, left, bottom = 640, 240, 40, 200
    bar = (width - left - 20) / len(counts)
    top_count = max(max(counts), 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             f'<rect width="{width}" height="{height}" fill="{PANEL}" rx="6"/>']
    for i, (label, count) in enumerate(zip(labels, counts)):
        h = count / top_count * (bottom - 30)
        bx = left + i * bar + 8
        colour = MUTED if label == "no warning" else "#3B82F6"
        parts.append(f'<rect x="{bx:.1f}" y="{bottom - h:.1f}" width="{bar - 16:.1f}" height="{h:.1f}" rx="3" fill="{colour}"/>'
                     f'<text x="{bx + (bar - 16) / 2:.1f}" y="{bottom - h - 6:.1f}" fill="{TEXT}" font-size="12" '
                     f'text-anchor="middle" {FONT}>{count}</text>'
                     f'<text x="{bx + (bar - 16) / 2:.1f}" y="{bottom + 18}" fill="{MUTED}" font-size="11" '
                     f'text-anchor="middle" {FONT}>{_esc(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _table(rows, header=None):
    head = "<tr>" + "".join(f"<th>{_esc(h)}</th>" for h in header) + "</tr>" if header else ""
    body = "".join("<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table>{head}{body}</table>"


def _pct(value):
    return "—" if value is None else f"{value}%"


# Write events/*.svg, lead_times.svg, summary.json and report.html into out_dir.
def write_report(days, summary, out_dir):
    events = sorted((event for day in days for event in day["events"]),
                    key=lambda e: (e["lead_s"] is None, -(e["lead_s"] or 0)))
    os.makedirs(os.path.join(out_dir, "events"), exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "events", "*.svg")):  # only this run's emergencies
        os.remove(old)
    cards = []
    for event in events:
        name = f"events/{event['date']}_{event['icao24'].replace('~', '_')}.svg"
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(event_svg(event))
        cards.append(f'<figure><img src="{_esc(name)}" alt="{_esc(event["flight_id"])} emergency"></figure>')
    with open(os.path.join(out_dir, "lead_times.svg"), "w", encoding="utf-8") as f:
        f.write(lead_histogram_svg(events))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    quartiles = summary["lead_quartiles_s"]
    headline = [
        ("Warned before the emergency", f"{summary['warned']} of {summary['events']}",
         f"{_pct(summary['warned_pct'])} · 95% range {summary['warned_pct_95'][0]}–{summary['warned_pct_95'][1]}%"
         if summary["warned_pct_95"] else _pct(summary["warned_pct"])),
        ("Median warning lead", format_duration(summary["lead_median_s"]), "among warned emergencies"),
        ("Warned 5+ minutes ahead", _pct(summary["warned_5min_pct"]), "of all emergencies"),
        ("Same check on normal flights", _pct(summary["chance_warned_pct"]),
         f"chance level · emergencies are {summary['lift']}× that" if summary["lift"] else "chance level"),
        ("Normal aircraft flagged", _pct(summary["sample_flagged_pct"]),
         f"{summary['sample_onsets_per_hour']} warnings per flight hour"),
    ]
    by_warning = [(row["cause"], _pct(row["before_emergencies_pct"]), _pct(row["normal_moments_pct"]),
                   "—" if row["lift"] is None else f"{row['lift']}×") for row in summary["by_warning"]]
    tiles = "".join(f'<div class="tile"><div class="label">{_esc(a)}</div><div class="value">{_esc(b)}</div>'
                    f'<div class="note">{_esc(c)}</div></div>' for a, b, c in headline)
    filters = days[0].get("filters", {})
    ignored = []
    if filters.get("takeoff_s") or filters.get("landing_s"):
        ignored.append(f"first {filters['takeoff_s'] / 60:g} min after takeoff and last "
                       f"{filters['landing_s'] / 60:g} min before landing (except near-ground overspeed)")
    if filters.get("min_warning_s"):
        ignored.append(f"orange that clears in under {filters['min_warning_s']:g} s")
    detail_rows = [
        ("Days", ", ".join(summary["days"])),
        ("Area", days[0]["area"]),
        ("Warnings ignored", "; ".join(ignored) or "none"),
        ("Aircraft seen", f"{summary['aircraft']:,}"),
        ("Emergencies (airborne, lasting ≥ {} reports)".format(MIN_RED_REPORTS), summary["events"]),
        ("Not counted", ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in summary["excluded"].items()) or "none"),
        ("Warned at least 1 minute ahead", _pct(summary["warned_1min_pct"])),
        ("Lead time, mean", format_duration(summary["lead_mean_s"])),
        ("Lead time, middle half", " – ".join(format_duration(q) for q in (quartiles[0], quartiles[2]))
         if quartiles else "—"),
        ("Lead time, longest", format_duration(summary["lead_max_s"])),
        ("Normal aircraft sampled", f"{summary['sample_aircraft']:,}"),
    ]
    per_day = [(day["date"], f"{day['aircraft']:,}", len(day["events"]),
                sum(1 for e in day["events"] if e["lead_s"] is not None)) for day in days]

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Syren early warning</title>
<style>
  :root {{ --bg: {BACKGROUND}; --panel: {PANEL}; --grid: {GRID}; --text: {TEXT}; --muted: {MUTED}; }}
  body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 "IBM Plex Sans", system-ui, sans-serif; }}
  main {{ max-width: 1000px; margin: 0 auto; padding: 24px 16px 64px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }} h2 {{ font-size: 15px; margin: 32px 0 10px; color: var(--muted); font-weight: 500; }}
  .sub {{ color: var(--muted); margin: 0 0 20px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
  .tile {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 8px; padding: 14px; }}
  .label, .note {{ color: var(--muted); font-size: 12px; }} .value {{ font-size: 26px; font-weight: 600; margin: 4px 0; }}
  table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
  td, th {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); }} th {{ color: var(--muted); font-weight: 500; }}
  figure {{ margin: 0 0 16px; }} img {{ max-width: 100%; height: auto; border-radius: 8px; display: block; }}
  .notes li {{ color: var(--muted); margin-bottom: 6px; }}
  @media (max-width: 600px) {{ .value {{ font-size: 20px; }} }}
</style></head>
<body><main>
<h1>Syren early warning</h1>
<p class="sub">How long before an aircraft turned red (emergency squawk) the app already showed it yellow
(prediction) or orange (detector). Generated {datetime.datetime.now():%Y-%m-%d %H:%M}.</p>
<div class="tiles">{tiles}</div>
<h2>Details</h2>{_table(detail_rows)}
<h2>Lead times</h2><img src="lead_times.svg" alt="Bar chart of warning lead times">
<h2>Which warnings mean something</h2>
<p class="sub">For each kind of warning: how often it appeared in the {LOOKBACK_S // 60} minutes before an emergency,
and how often the same window held it at an ordinary moment of a normal flight. A lift well above 1× means
the warning is more common before emergencies; near 1× means it fires about as often anyway.</p>
{_table(by_warning, ("Warning", "Before emergencies", "Ordinary moments", "Lift"))}
<h2>What gave the first warning</h2>{_table(summary["first_warning_by"].items(), ("Warning", "Emergencies"))}
<h2>Warnings on normal aircraft (the cost of early warning)</h2>
{_table(summary["sample_by_cause"].items(), ("Warning", "Times it started"))}
<h2>Per day</h2>{_table(per_day, ("Date", "Aircraft", "Emergencies", "Warned early"))}
<h2>Emergencies, most warned first</h2>{"".join(cards) or "<p class='sub'>No emergencies found.</p>"}
<h2>How this is measured</h2>
<ul class="notes">
<li>Each saved day's traces go through the same parser, detection engine and warning engine as a replay,
one aircraft at a time, and every report gets the map's colour: red beats yellow beats orange beats green.</li>
<li>An emergency is the first time an airborne aircraft turns red (squawk 7500/7600/7700, an emergency status,
or the detector's squawk alert) and stays red for {MIN_RED_REPORTS} reports. Lifeguard-only status, on-ground squawks
and one-report blips are not counted.</li>
<li>A warning counts if the aircraft was yellow or orange within {LOOKBACK_S // 60} minutes before turning red, with no
reception gap over {MAX_GAP_S // 60} minutes in between. The lead time is from the earliest such report.</li>
<li>Normal aircraft are a fixed sample (by ICAO address) of aircraft that never report an emergency that day;
"warnings per flight hour" counts every time one of them turned yellow or orange.</li>
<li>Filtered warnings (see "Warnings ignored") are turned back to green before anything is measured, for
emergencies and normal flights alike. A takeoff or landing is an airborne stretch that starts or ends next to an
on-ground report or below 3,000 ft. Yellow is never length-filtered: a prediction alert is one report by design.
Ignoring the landing phase uses hindsight; the app can't know live that a landing is coming.</li>
<li>The chance level checks those normal flights every 5 minutes of airborne time with exactly the same
window as an emergency. If it is close to the emergency figure, being "warned early" says little on its own.</li>
<li>Aircraft are evaluated on their own, so the conflict detector (which compares aircraft) doesn't run here.
Emergencies are defined by what the transponder reported, not by confirmed incidents.</li>
</ul>
</main></body></html>
"""
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(page)
    return summary
