# Syren

Run the frontend and backend in separate terminals from the repository root.

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
and serves the history API on port 8000; Vite proxies `/api` requests there.
