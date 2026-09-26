# ADS-B collection and labels

## Runtime

Use **CPython 3.13.3** (also recorded in the repository's `.python-version`). The data collectors use only the Python standard library; no `requirements.txt`, virtual environment, or package installation is needed. Model-training libraries belong in a later, separate training environment once the collected data and target have been audited.

## Whole-day corpus collection

ADSB.lol publishes daily aircraft trace archives through its `globe_history_YYYY` GitHub repositories. A full global day is commonly about 2.5–3.5 GiB compressed, so collection is deliberately date-selected and streams the archive without extracting every member. The collector keeps every trace with an explicit emergency declaration or emergency squawk, plus a deterministic random sample of other aircraft traces.

Example, from the repository root:

```powershell
python scripts/collect_adsb_days.py --dates 2024-02-15 2024-05-15 2024-08-15 2024-11-15 2025-02-15 2025-05-15 2025-08-15 2025-11-15
```

Outputs are local and git-ignored under `data/collection/`:

- `traces/YYYY-MM-DD/emergency/`: complete raw aircraft-day traces matching explicit emergency fields or squawks 7500, 7600, or 7700.
- `traces/YYYY-MM-DD/control_sample/`: complete raw traces from a repeatable uniform sample of other aircraft-day traces.
- `emergency_candidates/YYYY-MM-DD.jsonl`: candidate identity, signal type/value, point offset, source archive member, and trace path.
- `control_manifests/YYYY-MM-DD.jsonl`: aircraft metadata and path for each sampled trace.
- `reports/YYYY-MM-DD.json`: scan counts and archive URLs/checksum for reproducibility.

The archive is temporary and deleted after each scan by default. `--keep-archives` retains it; `--max-archive-bytes` defaults to 6 GiB and protects against an unexpected download. Re-running a date replaces its manifests and selected traces, making the date-level result repeatable. For example, retain 250 controls per date with `--controls-per-day 250`.

ADSB.lol documents that the archive contains one gzipped JSON aircraft-day trace per aircraft and that historical data is offered under the ODbL 1.0 license. Preserve source dates, URLs, archive checksums, and attribution when sharing derived databases. [ADSB.lol historical data documentation](https://www.adsb.lol/docs/open-data/historical/).

## Label policy and model target

An `observed_adsb_emergency_signal` is a weak label grounded in an explicit transponder emergency value or emergency squawk. The source trace itself is retained so the signal, onset, and aircraft context can be reviewed. These indications can be absent, delayed, or erroneous; they do not establish a real-world emergency independently.

`unlabeled_no_observed_declaration` is a sampled trace with no matching signal. It is **not** a verified normal-flight negative. Controls are sampled aircraft-days, not balanced by event, region, phase of flight, aircraft type, or ADS-B coverage. They are useful for exploratory review and initial feature-pipeline validation, not yet for calibrated risk or performance claims.

The catalog in `events.json` separately records reported accident and in-flight emergency outcomes. Outcome involvement, ADS-B declaration, and precursor-to-event are different targets. Do not label every point from an event aircraft as positive. The separate declaration and verified-incident labelers, candidate audit, model comparison, and alert replay are documented in [`ml/README.md`](../ml/README.md). Incident controls remain weak negatives until independently reviewed; ambiguous and missing cases are retained in the coverage report rather than silently assigned negative labels.

For a serious predictive dataset, expand collection across randomly selected dates, geographies, seasons, and years; add independently verified incident, accident, diversion, and emergency-report sources; deduplicate aircraft-days and events; review false/missed ADS-B declarations; and split evaluation by date/event/aircraft to prevent leakage. The day sample is a collection pilot, not evidence that ADS-B alone can predict emergencies.

## Event-anchored per-aircraft fetch

The original small outcome-seed fetcher remains available for inspecting aircraft involved in `events.json`:

```powershell
python scripts/fetch_event_traces.py
python scripts/fetch_event_traces.py --offline
```

It saves source files under `data/raw/adsb_lol/` and normalized observations/reports under `data/processed/`. A missing aircraft trace is a coverage gap, never a negative label. Some accident aircraft may not broadcast ADS-B Out.
