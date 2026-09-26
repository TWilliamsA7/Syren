# Syren

Run the backend and frontend in separate terminals from the repository root.

Terminal 1:

```sh
python -m backend.server
```

Terminal 2:

```sh
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal. The backend starts the live ADS-B feed
and serves the history and Gemini APIs on port 8000; Vite proxies `/api` there.
The live feed runs through the anomaly detector before the aircraft API serves
it, and detected aircraft are shown red on the map. To verify the map with a
deterministic anomaly showcase instead of live ADS-B, run this in Terminal 1:

```sh
python -m backend.demo_server --speed 20
```
