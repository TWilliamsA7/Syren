# Syren

```sh
python3 -m simulation.generate data/generated/train_001 --flights 1000 --seed 1
python3 -m detection.evaluate_generated data/generated/train_001
```

## Run the end-to-end demo

The local demo runs the anomaly showcase continuously, sends every simulated
state through the detection engine, and serves the latest enriched aircraft
snapshot to the browser. Start these from the repository root in two terminals.

Terminal 1, start the simulation and detector API:

```sh
python -m backend.demo_server --speed 20
```

Terminal 2, install frontend dependencies once and start Vite:

```sh
cd frontend
npm ci
npm run dev
```

Open the URL printed by Vite. The map refreshes once per second; detected
anomalies and emergency squawks turn aircraft red. The default scenario repeats
when it reaches the end. Use `--speed 1` for real-time simulation or pass
`--scenario path/to/scenario.json` to select another scenario. Stop either
process with Ctrl+C.
