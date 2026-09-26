# Trainable ADS-B declaration pipeline

Use **CPython 3.13.3**. The labeler and exported model inference use only the standard library. Local baseline training uses the pinned dependency in `requirements.txt`.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ml.label_windows
.\.venv\Scripts\python.exe -m ml.train_baseline
```

The labeler reads `data/collection/` without network access. It rechecks each manifest against the raw aircraft trace, splits at flight boundaries and observation gaps, and creates a 5-minute kinematic history at 60-second strides. A positive row has the **first observed ADS-B emergency declaration** 2–10 minutes after its anchor. A negative row has no declaration within the next 10 minutes and enough continuous trace coverage to observe that interval. Rows too close to the declaration, near a data gap, or short on history are excluded. The feature set omits emergency fields, squawk, identity, callsign, date, and coordinates.

Outputs under `data/learning/`:

- `declaration_windows.jsonl`: labeled examples, grouping keys, provenance, and numeric features.
- `declaration_windows_summary.json`: target definition, feature order, configuration, counts, and excluded candidates.
- `model/baseline_model.json`: standardized logistic model coefficients and feature schema, readable by `ml.predict.probability` with only standard Python on the Jetson.
- `model/holdout_metrics.json` and `model/holdout_predictions.jsonl`: evaluation on the latest collected date. Training excludes that date and also removes any aircraft that appears in the holdout.

The initial three-day cohort is **enriched with candidate aircraft** and contains many correlated windows per aircraft. Holdout metrics measure this development sample only; the model score is a ranking score, not a calibrated emergency probability or an operational alert threshold. More dates, independently confirmed incident outcomes, stronger matched controls, and event-level review are required before using it as an alerting model.

To predict independently confirmed real-world incidents, create a separate verified event manifest with event identity, aircraft, UTC event time, time precision, outcome type, and source. The current target is a future *declaration* and must remain distinct from that outcome target.
