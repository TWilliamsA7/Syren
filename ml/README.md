# Emergency prediction research pipeline

Use **CPython 3.13.3**. Training runs on the laptop; label generation and the
exported logistic model's inference require only Python's standard library.
The current data are three sampled 2024 days plus three separate accident traces.
No command below downloads ADS-B archives.

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ml.audit_candidates
.\.venv\Scripts\python.exe -m ml.label_windows
.\.venv\Scripts\python.exe -m ml.compare_models
.\.venv\Scripts\python.exe -m ml.bootstrap_report
.\.venv\Scripts\python.exe -m ml.train_research_model
.\.venv\Scripts\python.exe -m ml.label_incidents
.\.venv\Scripts\python.exe -m ml.train_baseline
```

## Declaration proxy

The audit checks every retained candidate against its raw aircraft trace and
records signal values, onset, coverage, and decision in
`data/learning/candidate_audit.jsonl`. Its decisions are **automated trace
checks**, not independent confirmation of emergencies. A `lifeguard`-only
priority status is excluded from the aircraft-distress proxy because it can
identify a priority medical transport rather than distress aboard that aircraft
([FAA guidance](https://www.faa.gov/sites/faa.gov/files/ERAS%20Flight%20Plan%20Interface%20Reference%20Guide%20%282022-10-01%29.pdf)).
Optional human decisions can be supplied to
`ml.audit_candidates --reviews path/to/reviews.jsonl`; each JSONL row needs
`date_utc`, `icao24`, `decision` (`include_proxy` or `exclude_proxy`),
`reviewer`, and `reason`.

The declaration labeler requires the audit. Each example has five minutes of
history and an anchor at least two minutes before a first usable declaration.
A positive has onset within the following 2–10 minutes; a negative has no
declaration within a continuously observed 10-minute future. It excludes
short histories, flight-leg boundaries, observation gaps, and insufficient
future coverage. Emergency status, squawk, identity, callsign, date, and raw
coordinates are never model features. The file contains both the original
17-value `features` and a causal `temporal_features` set with 30-, 60-,
180-, and 300-second trends, flight phase, and observation quality.

`ml.compare_models` runs aircraft-disjoint leave-one-date-out development
folds. It compares unweighted and normalized aircraft-day-balanced losses
with 2× and 4× positive variants, regularized logistic regression, and
constrained histogram gradient boosting on both feature sets. Outputs in
`data/learning/comparison/` contain every configuration, per-date train and
validation metrics, the best exploratory out-of-fold scores, and an alert
curve. All three dates are **development** data: August 2024 has already been
inspected, and selecting a configuration or threshold on these folds does not
give an untouched test estimate. The command names those three dates by default
and ignores any subsequently collected dates unless `--development-dates` is
explicitly changed. The conservative threshold caps raw control
window exceedances at one per 1,000 eligible observed airborne hours in sampled
controls; the current control exposure is only about 516 hours. Replay applies
a 10-minute alert suppression period. `ml.bootstrap_report` resamples whole
aircraft-days within dates to give descriptive intervals for the selected
development configuration. `ml.train_research_model` refits that selected
configuration on all three days and saves a research-only `joblib` artifact;
logistic selections also get a portable JSON export. The artifact requires
the pinned scikit-learn runtime for CPU inference if it is a boosted tree.

## Verified outcomes

`ml.label_incidents` reads the source-backed `data/events.json` registry
and existing cached accident traces. It produces
`data/learning/verified_incident_windows.jsonl` and a coverage summary.
Event-time precision is propagated into the 2–10 minute label: any window
whose class changes within the timestamp uncertainty is excluded. Its
aircraft-type- and flight-phase-matched controls are labeled
`no_registry_match_weak_negative`, with review status
`not_independently_cleared`. They must not be described as confirmed normal
flights. This target is separate from ADS-B declaration prediction.

Two of the three locally available accident aircraft have eligible positive
windows. This is sufficient to exercise the labeler, not to train or
evaluate a verified-emergency classifier. The NTSB's [aviation
database](https://www.ntsb.gov/Pages/AviationQueryV2.aspx) and the FAA's
[accident and incident data](https://www.faa.gov/av-info) are discovery
sources; a discovered case enters `data/events.json` only after its aircraft,
UTC event time, precision, event type, and source have been checked.
An official NTSB aviation CSV export can be staged with
`python -m ml.discover_events --ntsb-csv path/to/export.csv`. The resulting
`data/learning/ntsb_event_candidates.jsonl` is a review queue, never a
training label. Event dates in the export are not assumed to be UTC.

## Deployment and acceptance

The current three-day, 56-configuration development sweep selected a 3-leaf,
100-iteration temporal boosted tree by the specified alert rule. It has
out-of-fold ROC AUC 0.490, average precision 0.011, and detects 1 of 41
declaration events at a threshold with zero observed control alerts.
Only 516 eligible control airborne hours were observed, so the 95% Poisson
upper bound remains 7.15 false alerts per 1,000 hours. The research model
is **not accepted for operational alerting**; the larger train-versus-date
validation gap indicates poor generalization. Full results and aircraft-day
bootstrap intervals are under `data/learning/comparison/`.

`ml.train_baseline` exports a JSON logistic model under
`data/learning/model/`. `ml.predict.probability` scores it without
scikit-learn and checks export parity during training. The Orin Nano can run
this small model on CPU. After copying the research artifact and representative
window data to the Orin, run `python -m ml.benchmark_inference --model PATH
--dataset PATH` there to record CPU p50/p95/p99 latency. No score is a
calibrated probability of a real-world
emergency. Before any alerting claim, select the model and threshold on
validation data, then replay on newly collected untouched dates and report
event recall, warning time, and false alerts per 1,000 observed flight-hours
with uncertainty from independent flights or events.
