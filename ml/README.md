# Emergency prediction research pipeline

Use **CPython 3.10.0** from `C:\tools\Python310_Custom\python.exe` for the
local research environment. Legacy logistic inference remains standard
library-only; the primary Isolation Forest predictor uses the pinned
scikit-learn runtime. The ten-day declaration corpus is development data; it
does not include an untouched final test date. No command below downloads ADS-B
archives.

From the repository root, install the pinned dependencies and run the explicit
ten-day commands below. The legacy three-day artifacts remain available for
reference; these commands avoid overwriting them.

The ten-day experiment writes into its own ignored research directory and keeps
the original three-day outputs intact:

```powershell
.\.venv\Scripts\python.exe -m ml.audit_candidates --output-dir data/learning/ten_day_development
.\.venv\Scripts\python.exe -m ml.label_windows --episode-target --output-dir data/learning/ten_day_development --audit-file data/learning/ten_day_development/candidate_audit.jsonl
.\.venv\Scripts\python.exe -m ml.compare_temporal_models --dataset data/learning/ten_day_development/declaration_episode_windows.jsonl --output-dir data/learning/ten_day_development/temporal_episode_comparison
.\.venv\Scripts\python.exe -m ml.diagnose_temporal_signal
```

Score one previously extracted `temporal_features` object with the research
bundle:

```powershell
.\.venv\Scripts\python.exe -m ml.predict --model data/learning/ten_day_development/temporal_episode_comparison/isolation_forest_research_bundle.joblib --features-json path/to/temporal_features.json
```

The result field is `anomaly_score`; a larger value means the window is more
unusual relative to sampled controls. It is not an emergency probability.

`compare_temporal_models` makes a control-flight Isolation Forest the primary
predictor and compares it with the fixed 3-leaf/100-iteration temporal
histogram boosted-tree baseline. It uses ten leave-one-date-out folds and
removes aircraft IDs shared with each validation date from training. Each
forest is fit only on sampled control-flight windows, capped at four evenly
spaced windows per aircraft-day. These controls are not independently verified
normal flights. The forest scores the existing causal temporal
features from the preceding five minutes. Larger scores mean more unusual
behavior; they are not probabilities. Emergency signals, labels, subtype
values, absolute coordinates, and aircraft identity are never model inputs.

Signals within 60 seconds form one multi-label episode. Evaluation measures
any-declaration event recall, warning time, and false alerts per 1,000 observed
control airborne hours, with window ROC AUC and average precision secondary.
The report includes the full per-model score curves and a descriptive
comparison at no more than one false alert per 1,000 control hours. This
comparison does not select an operational threshold.

`diagnose_temporal_signal` compares event-level feature summaries against
sampled controls and candidate-aircraft negative segments, then reports
train-versus-date-validation fit for the boosted-tree reference and the
control-only forest on one fixed fold.

This workflow writes `declaration_episode_windows.jsonl` and its own summary;
the earlier `declaration_windows.jsonl` artifact is left intact. The output
includes fold scores, per-date curves, subtype counts, and nominal
Wilson/Poisson uncertainty intervals. The pooled curve is saved as
`event_recall_vs_false_alerts.svg`. The final all-date fit is saved as
`isolation_forest_research_bundle.joblib`; `ml.predict` can score one temporal
feature mapping with this bundle and labels the output as an anomaly score. The
bundle is research-only. No operational threshold is selected from these ten
dates; new dates collected after the model and threshold are frozen are needed
for a final evaluation.

## Declaration proxy

The audit checks every retained candidate against its raw aircraft trace and
records signal values, onset, coverage, and decision in the output directory's
`candidate_audit.jsonl`. Its decisions are **automated trace
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
coordinates are never model features. The file contains the original
17-value `features`, causal `temporal_features`, and (for the ten-day experiment)
the 31-step sequence described above.

## Historical supervised baseline results

The figures below are from the earlier tree/CNN experiments and do not report
the Isolation Forest. Rerun the documented ten-day workflow to create the
current forest comparison and research bundle.

The audit accepted 290 candidate traces under the ADS-B declaration proxy and
excluded 58 lifeguard/reserved-only traces. The labeler produced 1321 positive
windows from 209 eligible declaration events and 136,691 covered negative
windows. Another 81 included candidate traces had no eligible pre-event window:
75 lacked 420 seconds of leg history before declaration and 6 had fewer than
ten points in the matching leg. They are listed in the coverage summary and
not counted as evaluated events. The sampled controls contributed about 1,729
observed airborne hours.

The fixed temporal boosted tree had OOF ROC AUC 0.492 and average precision
0.00965; the CNN had 0.419 and 0.00794, against a 0.00957 positive-window
fraction. Event-level results were similarly weak: both detected zero events
at zero observed false alerts; the CNN's first detection was 1 of 209 events at
5.20 false alerts per 1,000 control airborne hours (warning time 572 seconds),
while the tree's first detection was 1 of 209 at 28.33 per 1,000 hours (196
seconds). The corresponding nominal 95% Poisson intervals are 2.38–9.88 and
20.96–37.46 false alerts per 1,000 hours; event recall for 1/209 has a Wilson
interval of 0.08%–2.66%. These operating points are curve descriptions, not
selected thresholds. The full curves and date-level scores are in
`data/learning/ten_day_development/temporal_comparison/`.

The CNN did not improve the ranking or event tradeoff consistently. Across
dates its ROC AUC ranged from 0.259 to 0.480; the tree ranged from 0.310 to
0.573. The next useful work is to inspect why the 81 candidate traces lack
eligible lead windows and review the 208 missed eligible events and their
coverage/trajectory patterns. These results do not support expanding the model
sweep or making an alerting claim. All ten dates were development data.

The follow-up signal audit found similar sequence coverage for positives and
controls (about 91% valid measurement slots). Its strongest univariate feature
was vertical-rate availability (event-versus-control flight-segment AUC 0.61),
while the strongest actual kinematic features were weak and often ranked in the
opposite direction. Aggregating out-of-fold scores by event and control flight
segment gave descriptive AUC 0.54 for the tree and 0.47 for the CNN. This
points away from missing samples as the main cause and toward weak kinematic
precursors, date/selection effects, and a target that combines different
actions: 89 `general` declarations, 71 `nordo`, 61 squawk 7700, 39 squawk
7600, with overlapping subtype counts. The 7500 subtype appears once. The
summary and feature-level details are in
`data/learning/ten_day_development/temporal_signal_diagnostics.json`; these
grouped statistics are descriptive, not a new operating metric.

On the fixed `2025-08-15` diagnostic fold, the tree's in-sample window ROC AUC
was 0.648 and fell to 0.482 on the held-out date; its average precision fell
from 0.0177 to 0.0152 (the held-out positive fraction was 0.0157). The CNN
went from ROC AUC 0.587 to 0.439, and its held-out average precision was also
at the 0.0157 random-ranking baseline. This single fold is not a final
estimate, but with the poor ten-date results it points to date/label
generalization and weak common precursors, rather than a model that simply
needs more layers.

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
The report also includes a nested threshold diagnostic: for each outer date,
the threshold is selected using cross-fitted scores from only the other two
dates, then replayed on the outer date. This reduces threshold leakage, but
the winning configuration is still selected on all three dates and each
threshold is based on only two inner dates.

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

The earlier three-day, 56-configuration development sweep selected a 3-leaf,
100-iteration temporal boosted tree by the specified alert rule. It has
out-of-fold ROC AUC 0.490, average precision 0.011, and detects 1 of 41
declaration events at a threshold with zero observed control alerts.
Under the nested threshold diagnostic, the fixed selected configuration
detected 0 of 41 events (0% recall), with no warning times to summarize and
0 false alerts across the same control exposure. The zero count still has a
95% Poisson upper bound of 7.15 per 1,000 hours. This is a development
diagnostic, not a final test estimate.
Only 516 eligible control airborne hours were observed, so the 95% Poisson
upper bound remains 7.15 false alerts per 1,000 hours. The research model
is **not accepted for operational alerting**; the larger train-versus-date
validation gap indicates poor generalization. Full results and aircraft-day
bootstrap intervals are under `data/learning/comparison/`.

`ml.train_baseline` exports a JSON logistic model under
`data/learning/model/`. `ml.predict.probability` scores it without
scikit-learn and checks export parity during training. No score is a calibrated
probability of a real-world emergency. Before any alerting claim, select the
model and threshold on validation data, then replay on newly collected
untouched dates and report event recall, warning time, and false alerts per
1,000 observed flight-hours with uncertainty from independent flights or
events. Device benchmarking is outside this experiment.
# Stateful behavior warning pilot

`python -m ml.temporal_risk_pilot` replays a fixed, causal evidence score on
2025-08-15. It uses altitude, vertical-rate, speed, track, and observation-gap
features; declaration status and squawk are excluded from the advance-warning
score. Evidence decays with time and needs at least two distinct behavior types.
The cutoff is chosen only from other dates' control flights under a budget of
25 suppressed false alerts per 1,000 observed control hours. This is a research
pilot, not the live detection threshold.

The pilot detected **1 of 38** declaration events 244 seconds early, with
**7 false alerts in 172.9 control hours** (40.5 per 1,000 hours). Its gate was
at least 3 events on at least 2 aircraft and at most 4 false alerts. It failed,
so the stateful score was not connected to the live warning path. The full
result is written to
`data/learning/ten_day_development/temporal_risk_pilot/pilot_2025-08-15.json`.
These development data cannot establish future-date warning performance.

# Positive-first pattern pilot

`python -m ml.positive_pattern_pilot` makes a 214-event casebook from the raw
ten-minute pre-declaration traces, excluding squawk and emergency status from
the behavioral timeline. It discovers simple, named patterns from positive
windows on nine dates, requiring recurrence across at least five events, five
aircraft, and three dates. The 2026-05-15 aircraft are purged from discovery.
The frozen patterns are replayed on that one date with ten-minute alert
suppression. No fixed false-alert budget or live threshold is selected.

The lowest-alert individual pattern in this pilot, altitude reversal, warned
8 of 24 events a median of 175 seconds before declaration, with 127 alerts in
182.7 control flight-hours (695 per 1,000 hours, about one per 1.4 hours).
At approximately the same false-alert rate, the previous Isolation Forest
warned 2 events and the boosted tree warned 1. Adding patterns caught up to
20 events but raised the control alert rate to 6,042 per 1,000 hours. The
patterns were therefore kept research-only. Four events were still missed by
that broad union, including radio-failure and general-status declarations.

The full per-pattern results, subtype counts, matched baseline points,
missed-event IDs, and control alert examples are in
`data/learning/ten_day_development/positive_pattern_pilot/pilot_2026-05-15.json`.
The raw casebook is in the same directory. All ten dates are development data,
and repeated exploratory use of them limits generalization claims.

# Context-matched pattern follow-up

`python -m ml.contextual_pattern_pilot` reviews the 127 control alerts from the
prior pilot, then scores each named behavior by its rarity among training
controls in a comparable flight phase, altitude band, speed band, and aircraft
type when enough controls exist. It samples control windows per aircraft-day,
purges held-out aircraft, discovers cutoffs using positive event peaks on nine
dates, and replays 2024-11-15 once. The pilot reports each pattern and their
union without imposing a false-alert cap. Its output is
`data/learning/ten_day_development/contextual_pattern_pilot/pilot_2024-11-15.json`.

The prior false alerts were mostly level-flight observations (83 of 127), with
a median reversal of only 125 feet. Context matching did **not** improve the
warning tradeoff: none of 27 candidate cutoffs reached twofold event enrichment
over matched control alerts on the training dates. On the new replay date, the
original 100-foot reversal rule warned 7 of 17 events with 129 control alerts
in 177.0 hours (729 per 1,000 hours); the context-matched patterns caught fewer
events at similar rates. This follow-up remains research-only. The weak
controls and repeated development-date analysis still prevent future-date
performance claims.
