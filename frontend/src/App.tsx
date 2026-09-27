import React, { useState, useEffect, useRef } from "react";
import DeckGL from "@deck.gl/react";
import Map from "react-map-gl/maplibre"; // or 'react-map-gl' for Mapbox
import { IconLayer } from "@deck.gl/layers";

import "./App.css";
import logo from "./assets/syren-logo.png";

interface Aircraft {
  hex: string;
  type: string;
  flight?: string;
  r?: string;
  t?: string;
  alt_baro?: number | string;
  alt_geom?: number;
  gs?: number;
  track?: number;
  baro_rate?: number;
  geom_rate?: number;
  squawk?: string;
  emergency?: string;
  category?: string;
  nav_qnh?: number;
  nav_altitude_mcp?: number;
  nav_heading?: number;
  lat?: number;
  lon?: number;
  nic?: number;
  rc?: number;
  version?: number;
  messages?: number;
  seen?: number;
  rssi?: number;
  dst?: number;
  dir?: number;
  timestamp?: string;
  detection?: {
    icao24: string;
    flight_id: string;
    timestamp: number;
    risk_score: number;
    severity: "normal" | "advisory" | "warning" | "critical";
    anomalies: { type: string; severity: number; message: string }[];
  };
  state?: Record<string, unknown>; // the full FlightState from the feed, sent to Gemini
  anomaly?: string; // set by the AI model, see docs/protocol.md; "none" when nothing is wrong
}

interface FlightStateFeed {
  icao24: string;
  flight_id?: string;
  aircraft?: {
    type_code?: string | null;
    registration?: string | null;
    category?: string | null;
  } | null;
  position?: {
    latitude?: number | null;
    longitude?: number | null;
    altitude_baro_ft?: number | string | null;
    altitude_geom_ft?: number | null;
  } | null;
  kinematics?: {
    ground_speed_kts?: number | null;
    track_deg?: number | null;
    vertical_rate_baro_fpm?: number | null;
  } | null;
  status?: { squawk?: string | null; emergency?: string | null } | null;
  detection?: Aircraft["detection"];
  anomaly?: string;
  timestamp?: number;
  [key: string]: unknown;
}

interface HistoryStatus {
  state: string;
  date: string | null;
  total_bytes: number;
  done_bytes: number;
  export_total: number;
  export_done: number;
  error: string | null;
}

// The aircraft the map is following: onlyThis hides every other plane,
// centre is switched off when the user drags the map away
interface Followed {
  hex: string;
  onlyThis: boolean;
  centre: boolean;
}

const FOLLOW_ZOOM = 8; // zoom in at least this far when starting to follow an aircraft

// Gemini's answer for one aircraft, kept with its hex so a late reply can't land on another plane
interface GeminiAnswer {
  hex: string;
  status: "loading" | "done" | "error";
  text: string;
}

const DEFAULT_VIEW_STATE = {
  longitude: -95.7129,
  latitude: 37.0902,
  zoom: 4,
  maxZoom: 18,
  minZoom: 2,
  pitch: 0,
  bearing: 0,
};

// Region specific view states for zooming buttons
const REGION_VIEWS = {
  US_ALL: {
    longitude: -95.7129,
    latitude: 37.0902,
    zoom: 4,
    pitch: 0,
    bearing: 0,
  },
  US_WEST: {
    longitude: -120.5583,
    latitude: 40.5556,
    zoom: 4.7,
    pitch: 0,
    bearing: 0,
  },
  US_MIDWEST: {
    longitude: -101.6298,
    latitude: 41.8781,
    zoom: 5,
    pitch: 0,
    bearing: 0,
  },
  US_SOUTH: {
    longitude: -93.797,
    latitude: 31.7767,
    zoom: 5.15,
    pitch: 0,
    bearing: 0,
  },
  US_EAST: {
    longitude: -75.1652,
    latitude: 39.9526,
    zoom: 5,
    pitch: 0,
    bearing: 0,
  },
};

// Inline SVG atlas: normal (green), detector anomaly (orange), emergency squawk (red).
// The icons aren't masks, so getColor can't recolor them; getIcon picks one.
const PLANE_PATH =
  "M12 2a1.5 1.5 0 0 1 1.5 1.5v5.25l7 3.75v1.75l-7-2.25v5l2 1.5v1.25l-3.5-1-3.5 1v-1.25l2-1.5v-5l-7 2.25v-1.75l7-3.75V3.5A1.5 1.5 0 0 1 12 2z";
const AIRPLANE_ICON = `data:image/svg+xml;charset=utf-8,<svg xmlns="http://www.w3.org/2000/svg" width="384" height="128" viewBox="0 0 72 24" stroke="black" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"><path fill="%2336F6B4" d="${PLANE_PATH}"/><path fill="%23FB923C" transform="translate(24 0)" d="${PLANE_PATH}"/><path fill="%23F43F5E" transform="translate(48 0)" d="${PLANE_PATH}"/></svg>`;
const ICON_MAPPING = {
  marker: { x: 0, y: 0, width: 128, height: 128, mask: false },
  detected: { x: 128, y: 0, width: 128, height: 128, mask: false },
  squawk: { x: 256, y: 0, width: 128, height: 128, mask: false },
};

const EMERGENCY_SQUAWKS = new Set(["7500", "7600", "7700"]);
type AlertKind = "squawk" | "detected" | null;

const alertKind = (aircraft: Aircraft): AlertKind => {
  const detectorAnomalies = aircraft.detection?.anomalies ?? [];
  const declaredEmergency =
    !!aircraft.emergency && aircraft.emergency !== "none";
  const emergencySquawk = EMERGENCY_SQUAWKS.has(
    aircraft.squawk?.trim() ?? "",
  );
  const detectorSquawk = detectorAnomalies.some(
    (anomaly) => anomaly.type === "EMERGENCY_SQUAWK",
  );
  if (
    declaredEmergency ||
    emergencySquawk ||
    detectorSquawk ||
    aircraft.anomaly === "squawk"
  ) {
    return "squawk";
  }
  if (
    detectorAnomalies.length > 0 ||
    (!!aircraft.anomaly && aircraft.anomaly !== "none")
  ) {
    return "detected";
  }
  return null;
};

type Region = "US_ALL" | "US_WEST" | "US_MIDWEST" | "US_SOUTH" | "US_EAST";
const REGION_LABELS: [Region, string][] = [
  ["US_ALL", "All US"],
  ["US_WEST", "West"],
  ["US_MIDWEST", "Midwest"],
  ["US_SOUTH", "South"],
  ["US_EAST", "East"],
];

// What the header shows for each replay state before the clock starts
const REPLAY_STATE_TEXT: Record<string, string> = {
  idle: "Starting…",
  loading: "Starting…",
  downloading: "Downloading archive…",
  exporting: "Building replay…",
  failed: "Failed",
  stopped: "Stopped",
};

const formatNumber = (
  value: number | string | null | undefined,
  unit: string,
) => (typeof value === "number" ? `${value.toLocaleString()} ${unit}` : "—");

// The history archive's first day (FIRST_DAY in data/history.py)
const FIRST_HISTORY_DAY = "2023-02-16";

// The newest day that can be replayed: today minus one, as YYYY-MM-DD in local time
const latestHistoryDay = () => {
  const day = new Date();
  day.setDate(day.getDate() - 1);
  return [
    day.getFullYear(),
    String(day.getMonth() + 1).padStart(2, "0"),
    String(day.getDate()).padStart(2, "0"),
  ].join("-");
};

// "16:00", "1600" or "9:05" -> "16:00" / "09:05"; null if it isn't a 24-hour time
const parseTime24 = (text: string) => {
  const match = text.trim().match(/^(\d{1,2}):?(\d{2})$/);
  if (!match || Number(match[1]) > 23 || Number(match[2]) > 59) return null;
  return `${match[1].padStart(2, "0")}:${match[2]}`;
};

function App() {
  const [aircraftList, setAircraftList] = useState<Aircraft[]>([]);
  const [isLive, setIsLive] = useState<boolean>(true);
  const [lastUpdated, setLastUpdated] = useState<string>("—");

  const [currentRegion, setCurrentRegion] = useState<Region>("US_ALL");
  const [viewState, setViewState] = useState(DEFAULT_VIEW_STATE);

  // Search state
  const [searchDate, setSearchDate] = useState<string>("2026-09-24");
  const [searchStartTime, setSearchStartTime] = useState<string>("16:00");

  // Find/follow one aircraft, in live or replay
  const [findQuery, setFindQuery] = useState<string>("");
  const [findError, setFindError] = useState<string>("");
  const [followed, setFollowed] = useState<Followed | null>(null);

  const [searchIcao] = useState<string>("");

  // History / Replay state
  const [historyMode, setHistoryMode] = useState<boolean>(false);
  const [historyStatusInfo, setHistoryStatusInfo] = useState<HistoryStatus>({
    state: "idle",
    date: null,
    total_bytes: 0,
    done_bytes: 0,
    export_total: 0,
    export_done: 0,
    error: null,
  });
  const [replayClockFormatted, setReplayClockFormatted] = useState<string>("");

  // View states
  const [searchError, setSearchError] = useState<string>("");
  const [hoveredAircraft, setHoveredAircraft] = useState<Aircraft | null>(null);
  const [selectedAircraft, setSelectedAircraft] = useState<Aircraft | null>(
    null,
  ); // pinned by a click
  const [, setSearchedAircraft] = useState<Aircraft | null>(null);
  const [geminiAnswer, setGeminiAnswer] = useState<GeminiAnswer | null>(null);

  const consecutiveFailuresRef = useRef<number>(0);
  const askingHexRef = useRef<string | null>(null); // the aircraft whose Gemini answer is being waited for

  // Helper function to safely parse either JSON array or NDJSON (JSON Lines)
  const parseJsonData = (text: string) => {
    try {
      return JSON.parse(text);
    } catch {
      // Fallback for NDJSON / JSON Lines format
      return text
        .trim()
        .split("\n")
        .filter((line) => line.trim().length > 0)
        .map((line) => JSON.parse(line));
    }
  };

  // Poll history_status if history mode is active
  useEffect(() => {
    let statusTimer: NodeJS.Timeout;
    let cancelled = false; // set on cleanup, so a fetch still in flight can't keep an old loop going
    if (historyMode) {
      const pollStatus = async () => {
        try {
          const res = await fetch("/api/history_status", { cache: "no-store" });
          if (res.ok) {
            const data = await res.json();
            if (cancelled) return;
            setHistoryStatusInfo(data);
            if (data.clock) {
              // e.g. "2026-09-24 16:05:32Z"
              setReplayClockFormatted(
                new Date(data.clock * 1000)
                  .toISOString()
                  .slice(0, 19)
                  .replace("T", " ") + "Z",
              );
            }
          }
        } catch (e) {
          console.error("Failed to fetch history status", e);
        }
        if (!cancelled) statusTimer = setTimeout(pollStatus, 1000);
      };
      pollStatus();
    }
    return () => {
      cancelled = true;
      clearTimeout(statusTimer);
    };
  }, [historyMode]);

  // Poll the backend aircraft API, or the JSONL file while replaying history.
  useEffect(() => {
    let timer: NodeJS.Timeout;
    let cancelled = false; // set on cleanup, so a fetch still in flight can't keep an old loop going

    const fetchData = async () => {
      try {
        const endpoint = historyMode ? "/history.jsonl" : "/api/aircraft";
        const response = await fetch(endpoint, { cache: "no-store" });
        if (!response.ok) throw new Error("Failed to fetch data file");

        const rawApiResponse = historyMode
          ? parseJsonData(await response.text())
          : await response.json();
        if (cancelled) return;

        if (!Array.isArray(rawApiResponse))
          throw new Error("Aircraft API response must be a JSON array");
        const parsedData: Aircraft[] = rawApiResponse.map((value: unknown) => {
          if (typeof value !== "object" || value === null)
            throw new Error("Aircraft API returned an invalid record");
          const item = value as FlightStateFeed;
          return {
            hex: item.icao24,
            flight: item.flight_id,
            type: item.aircraft?.type_code || "adsb_icao",
            t: item.aircraft?.type_code || undefined,
            r: item.aircraft?.registration || undefined,
            lat: item.position?.latitude ?? undefined,
            lon: item.position?.longitude ?? undefined,
            alt_baro: item.position?.altitude_baro_ft ?? undefined,
            alt_geom: item.position?.altitude_geom_ft ?? undefined,
            gs: item.kinematics?.ground_speed_kts ?? undefined,
            track: item.kinematics?.track_deg ?? undefined,
            baro_rate: item.kinematics?.vertical_rate_baro_fpm ?? undefined,
            squawk: item.status?.squawk ?? undefined,
            emergency: item.status?.emergency ?? undefined,
            detection: item.detection,
            anomaly: item.anomaly,
            category: item.aircraft?.category ?? undefined,
            timestamp: item.timestamp
              ? new Date(item.timestamp * 1000).toISOString()
              : new Date().toISOString(),
            state: item,
          };
        });

        setAircraftList(parsedData);
        setIsLive(true); // COOKED only when the fetch fails, not during every replay
        setLastUpdated(new Date().toISOString().slice(11, 19) + "Z");
        consecutiveFailuresRef.current = 0;

        // Schedule next standard check interval (1 second)
        timer = setTimeout(fetchData, 1000);
      } catch {
        if (cancelled) return;
        consecutiveFailuresRef.current += 1;

        if (consecutiveFailuresRef.current === 1) {
          timer = setTimeout(fetchData, 500);
        } else {
          setIsLive(false);
          timer = setTimeout(fetchData, 2000);
        }
      }
    };

    fetchData();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [historyMode]);

  // Backend API Call Handlers for History Controls
  const startHistory = async (date: string, startTime: string = "16:00") => {
    setReplayClockFormatted(""); // don't show the last replay's clock while this one loads
    try {
      const response = await fetch("/api/start_history", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ date, start: startTime }),
      });
      if (response.ok) {
        setHistoryMode(true);
      }
    } catch (e) {
      console.error("Error calling start_history", e);
    }
  };

  const stopHistory = async () => {
    try {
      await fetch("/api/stop_history", { method: "POST" });
      setHistoryMode(false);
      setHistoryStatusInfo((current) => ({ ...current, state: "idle" }));
    } catch (e) {
      console.error("Error calling stop_history", e);
    }
  };

  const skipTime = async (seconds: number) => {
    try {
      await fetch("/api/skip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seconds }),
      });
    } catch (e) {
      console.error("Error calling skip", e);
    }
  };

  // Handle Search Submission
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setSearchError("");

    if (!searchDate) {
      setSearchError("A date is required to query the archive.");
      return;
    }
    if (searchDate < FIRST_HISTORY_DAY || searchDate > latestHistoryDay()) {
      setSearchError(`Pick a day from ${FIRST_HISTORY_DAY} to ${latestHistoryDay()}.`);
      return;
    }

    const startTime = parseTime24(searchStartTime);
    if (!startTime) {
      setSearchError("Start time must be 24-hour HH:MM, e.g. 16:00.");
      return;
    }
    setSearchStartTime(startTime);

    // Trigger backend history replay session for the selected date and start time
    startHistory(searchDate, startTime);

    const query = searchIcao.trim().toLowerCase();
    if (query) {
      const found = aircraftList.find(
        (a) =>
          a.hex.toLowerCase() === query ||
          (a.r && a.r.toLowerCase() === query) ||
          (a.flight && a.flight.trim().toLowerCase() === query),
      );

      if (found) {
        setSearchedAircraft({ ...found, timestamp: searchDate });
        if (found.lat && found.lon) {
          setViewState((v) => ({
            ...v,
            longitude: found.lon!,
            latitude: found.lat!,
            zoom: 9,
          }));
        }
      }
    } else {
      setSearchedAircraft(null);
    }

  };

  const returnToLive = () => {
    stopHistory();
    setSearchError("");
  };

  // Zoom to an aircraft, pin its popup, and keep the map centred on it as new positions arrive
  const followAircraft = (plane: Aircraft, onlyThis: boolean) => {
    setFollowed({ hex: plane.hex, onlyThis, centre: true });
    setSelectedAircraft(plane);
    if (plane.lat != null && plane.lon != null) {
      setViewState((v) => ({
        ...v,
        latitude: plane.lat!,
        longitude: plane.lon!,
        zoom: Math.max(v.zoom, FOLLOW_ZOOM),
      }));
    }
  };

  // Select an alert and centre once without entering follow mode.
  const selectAircraft = (plane: Aircraft) => {
    setFollowed(null);
    setSelectedAircraft(plane);
    if (plane.lat != null && plane.lon != null) {
      setViewState((v) => ({
        ...v,
        latitude: plane.lat!,
        longitude: plane.lon!,
        zoom: Math.max(v.zoom, FOLLOW_ZOOM),
      }));
    }
  };

  // Find an aircraft in whatever is showing (live or replay) by callsign, ICAO hex or registration
  const findAircraft = (e: React.FormEvent) => {
    e.preventDefault();
    const query = findQuery.trim().toLowerCase();
    if (!query) return;
    const found = aircraftList.find(
      (a) =>
        a.hex.toLowerCase() === query ||
        a.flight?.trim().toLowerCase() === query ||
        a.r?.toLowerCase() === query,
    );
    if (found) {
      setFindError("");
      setFindQuery("");
      followAircraft(found, true);
    } else {
      setFindError(
        `${findQuery.trim()} isn't in the ${historyMode ? "replay" : "live feed"} right now.`,
      );
    }
  };

  const followedAircraft = followed
    ? (aircraftList.find((a) => a.hex === followed.hex) ?? null)
    : null;

  // While following, the map is centred on the aircraft's latest position (zoom stays the user's)
  const mapViewState =
    followed?.centre &&
    followedAircraft?.lat != null &&
    followedAircraft?.lon != null
      ? {
          ...viewState,
          latitude: followedAircraft.lat,
          longitude: followedAircraft.lon,
        }
      : viewState;

  const handleRegionChange = (region: Region) => {
    setCurrentRegion(region);
    setViewState((v) => ({
      ...v,
      ...REGION_VIEWS[region],
    }));
  };

  const emergencyAircraft = aircraftList.filter(
    (aircraft) => alertKind(aircraft) === "squawk",
  );
  const emergencyCount = emergencyAircraft.length;
  const detectedAnomalies = aircraftList
    .flatMap((aircraft) =>
      (aircraft.detection?.anomalies ?? []).map((anomaly) => ({
        aircraft,
        anomaly,
        riskScore: aircraft.detection?.risk_score ?? anomaly.severity,
      })),
    )
    .sort((a, b) => b.riskScore - a.riskScore);

  // The pinned aircraft with its newest data; its last known data if it has left the feed
  const pinnedAircraft = selectedAircraft
    ? (aircraftList.find((a) => a.hex === selectedAircraft.hex) ??
      selectedAircraft)
    : null;
  const popupAircraft = pinnedAircraft ?? hoveredAircraft;
  const pinnedAnswer =
    pinnedAircraft && geminiAnswer?.hex === pinnedAircraft.hex
      ? geminiAnswer
      : null;

  const askGemini = async () => {
    if (!pinnedAircraft) return;
    const hex = pinnedAircraft.hex;
    // Only store the reply if this aircraft is still the one being asked about
    const answer = (status: GeminiAnswer["status"], text: string) =>
      setGeminiAnswer((previous) =>
        previous?.hex === hex ? { hex, status, text } : previous,
      );

    setGeminiAnswer({ hex, status: "loading", text: "" });
    askingHexRef.current = hex;

    // Start the question, then check for the answer every second. One request held open while
    // Gemini thinks gets dropped by the browser now and then (Firefox does after a network change);
    // a dropped check just retries a second later.
    let job: string;
    try {
      const response = await fetch("/api/ask_aircraft", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ aircraft: pinnedAircraft.state }),
      });
      const data = await response.json();
      if (!response.ok)
        return answer(
          "error",
          data.error ?? `Request failed (${response.status})`,
        );
      job = data.job;
    } catch {
      return answer(
        "error",
        "Couldn't reach the Syren server. Is python -m backend.server running?",
      );
    }

    let droppedChecks = 0;
    while (askingHexRef.current === hex && droppedChecks < 5) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      try {
        const response = await fetch(`/api/ask_aircraft/${job}`, {
          cache: "no-store",
        });
        const data = await response.json();
        if (data.status === "done") return answer("done", data.answer);
        if (data.status === "error" || !response.ok)
          return answer(
            "error",
            data.error ?? `Request failed (${response.status})`,
          );
        droppedChecks = 0;
      } catch {
        droppedChecks += 1;
      }
    }
    if (askingHexRef.current === hex)
      answer(
        "error",
        "Lost contact with the Syren server while waiting for Gemini.",
      );
  };

  const replayState: string = historyStatusInfo.state;
  const canSkip = historyMode && replayState === "playing";
  const preparingReplay =
    historyMode && replayState !== "playing" && replayState !== "stopped";
  const downloadPercent =
    historyStatusInfo.total_bytes > 0
      ? Math.round(
          (historyStatusInfo.done_bytes / historyStatusInfo.total_bytes) * 100,
        )
      : 0;
  const exportPercent =
    historyStatusInfo.export_total > 0
      ? Math.round(
          (historyStatusInfo.export_done / historyStatusInfo.export_total) *
            100,
        )
      : null; // the backend doesn't report export progress yet

  const feedLabel = !isLive ? "Offline" : historyMode ? "Replay" : "Live";
  const feedDot = !isLive ? "red" : historyMode ? "accent" : "green";
  const feedDetail = !historyMode
    ? "US airspace"
    : replayState === "playing"
      ? replayClockFormatted
      : (REPLAY_STATE_TEXT[replayState] ?? replayState);

  // Define Deck.GL Layers for Aircraft visualization mapped to geo coordinates
  const layers = [
    new IconLayer({
      id: "aircraft-icon-layer",
      data: followed?.onlyThis
        ? aircraftList.filter((a) => a.hex === followed.hex)
        : aircraftList,
      iconAtlas: AIRPLANE_ICON,
      iconMapping: ICON_MAPPING,
      getIcon: (d: Aircraft) => alertKind(d) ?? "marker",
      getPosition: (d: Aircraft) => [d.lon ?? -95.7129, d.lat ?? 37.0902],
      getSize: 24,
      getAngle: (d: Aircraft) => -(d.track ?? 0),
      pickable: true,
      onHover: (info) => setHoveredAircraft((info.object as Aircraft) || null),
      onClick: (info) => {
        if (info.object) setSelectedAircraft(info.object as Aircraft);
      },
      updateTriggers: {
        data: [aircraftList, followed],
        getAngle: [aircraftList, followed],
        getIcon: [aircraftList, followed],
      },
    }),
  ];

  return (
    <div className="syren-container">
      <header className="syren-header">
        <div className="header-left">
          <div className="brand">
            <img src={logo} alt="" className="logo" />
            <h1 className="wordmark">SYREN</h1>
          </div>
          <div className="mode">
            <span className={`dot ${feedDot}`} />
            <span className="mode-label">{feedLabel}</span>
            <span className={`mode-detail ${canSkip ? "mono" : ""}`}>
              {feedDetail}
            </span>
            {historyMode && (
              <button className="link" onClick={returnToLive}>
                Back to live
              </button>
            )}
          </div>
        </div>
        <div className="header-readout mono">updated {lastUpdated}</div>
      </header>

      <div className="syren-workspace">
        <div
          className="map-placeholder"
          style={{ position: "relative", overflow: "hidden" }}
        >
          <div
            className="map-grid-bg"
            style={{ zIndex: 1, pointerEvents: "none" }}
          />

          {/* Deck.GL Canvas Integration with Geographic Coordinate mapping */}
          <DeckGL
            viewState={mapViewState}
            onViewStateChange={(e) => {
              setViewState(e.viewState as typeof DEFAULT_VIEW_STATE);
              // dragging the map stops re-centring on a followed aircraft, so it doesn't snap back
              if (e.interactionState?.isDragging)
                setFollowed((f) => (f?.centre ? { ...f, centre: false } : f));
            }}
            controller={true}
            layers={layers}
            style={{ position: "absolute", inset: "0", zIndex: "2" }}
          >
            <Map
              reuseMaps
              mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
            />
          </DeckGL>

          {popupAircraft && (
            <div
              className={`map-tooltip ${pinnedAircraft ? "pinned" : ""} ${pinnedAnswer?.status === "loading" ? "asking" : ""}`}
            >
              <div className="callsign">
                {popupAircraft.flight?.trim() || "Unknown"}
                <span className="mono">{popupAircraft.hex}</span>
                {pinnedAircraft && (
                  <button
                    className="close"
                    onClick={() => setSelectedAircraft(null)}
                    aria-label="Close"
                  >
                    ×
                  </button>
                )}
              </div>
              <dl className="kv">
                <dt>Altitude</dt>
                <dd>{formatNumber(popupAircraft.alt_baro, "ft")}</dd>
                <dt>Ground speed</dt>
                <dd>{formatNumber(popupAircraft.gs, "kt")}</dd>
                <dt>Squawk</dt>
                <dd>{popupAircraft.squawk || "—"}</dd>
              </dl>

              {pinnedAircraft ? (
                <div className="gemini">
                  {pinnedAnswer?.status === "done" && (
                    <p className="gemini-answer">{pinnedAnswer.text}</p>
                  )}
                  {pinnedAnswer?.status === "error" && (
                    <p className="form-error">{pinnedAnswer.text}</p>
                  )}
                  <button
                    className="btn"
                    onClick={askGemini}
                    disabled={pinnedAnswer?.status === "loading"}
                  >
                    {pinnedAnswer?.status === "loading"
                      ? "Asking Gemini…"
                      : pinnedAnswer?.status === "done"
                        ? "Ask Gemini again"
                        : "Ask Gemini about this aircraft"}
                  </button>
                </div>
              ) : (
                <div className="tooltip-hint">
                  Click the aircraft to pin this
                </div>
              )}
            </div>
          )}

          <div className="map-toolbar" style={{ zIndex: 20 }}>
            {REGION_LABELS.map(([region, label]) => (
              <button
                key={region}
                className={currentRegion === region ? "active" : ""}
                onClick={() => handleRegionChange(region)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <aside className="sidebar">
          <section className="panel-section">
            <h2>Find aircraft</h2>
            <form onSubmit={findAircraft} className="find-row">
              <label className="field">
                <input
                  type="text"
                  placeholder="Callsign, ICAO hex or registration"
                  aria-label="Callsign, ICAO hex or registration"
                  value={findQuery}
                  onChange={(e) => setFindQuery(e.target.value)}
                />
              </label>
              <button type="submit" className="btn">
                Find
              </button>
            </form>
            {findError && <p className="form-error">{findError}</p>}

            {followed && (
              <div className="following">
                <div className="following-line">
                  <span className={`dot ${followedAircraft ? "accent" : ""}`} />
                  <span>
                    {followed.centre && followedAircraft
                      ? "Following"
                      : "Found"}
                  </span>
                  <strong className="mono">
                    {followedAircraft?.flight?.trim() || followed.hex}
                  </strong>
                  <button className="link" onClick={() => setFollowed(null)}>
                    Stop
                  </button>
                </div>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={followed.onlyThis}
                    onChange={(e) =>
                      setFollowed({ ...followed, onlyThis: e.target.checked })
                    }
                  />
                  Only show this aircraft
                </label>
                {!followedAircraft ? (
                  <p className="hint">
                    Not in the {historyMode ? "replay" : "live feed"} right now.
                  </p>
                ) : (
                  !followed.centre && (
                    <button
                      className="link"
                      onClick={() =>
                        followAircraft(followedAircraft, followed.onlyThis)
                      }
                    >
                      Re-centre on it
                    </button>
                  )
                )}
              </div>
            )}
          </section>

          <section className="panel-section">
            <div className="section-head">
              <h2>Replay a day</h2>
              {historyMode && (
                <button className="link" onClick={returnToLive}>
                  Back to live
                </button>
              )}
            </div>

            <form onSubmit={handleSearch}>
              <div className="field-row">
                <label className="field">
                  <span>Date</span>
                  <input
                    type="date"
                    min={FIRST_HISTORY_DAY}
                    max={latestHistoryDay()}
                    value={searchDate}
                    onChange={(e) => setSearchDate(e.target.value)}
                  />
                </label>
                <label className="field">
                  <span>Start (UTC)</span>
                  <input
                    type="text"
                    className="time-input"
                    inputMode="numeric"
                    maxLength={5}
                    placeholder="16:00"
                    value={searchStartTime}
                    onChange={(e) => setSearchStartTime(e.target.value)}
                  />
                </label>
              </div>
              {searchError && <p className="form-error">{searchError}</p>}
              <button type="submit" className="btn primary">
                Load replay
              </button>
            </form>

            {historyMode && (
              <div className="transport">
                <button
                  className="btn"
                  onClick={() => skipTime(-300)}
                  disabled={!canSkip}
                >
                  −5 min
                </button>
                <span className="mono">
                  {canSkip ? replayClockFormatted.slice(11) : "--:--:--"}
                </span>
                <button
                  className="btn"
                  onClick={() => skipTime(300)}
                  disabled={!canSkip}
                >
                  +5 min
                </button>
              </div>
            )}
          </section>

          {preparingReplay && (
            <section className="panel-section">
              <h2>Preparing {historyStatusInfo.date ?? searchDate}</h2>

              <div
                className={`step ${replayState === "downloading" ? "active" : ""}`}
              >
                <div className="step-line">
                  <span>Download archive</span>
                  <span className="mono">
                    {replayState === "downloading"
                      ? `${downloadPercent}%`
                      : replayState === "loading"
                        ? ""
                        : historyStatusInfo.total_bytes > 0
                          ? "done"
                          : "on disk"}
                  </span>
                </div>
                <div className="bar">
                  <div
                    style={{
                      width: `${replayState === "downloading" ? downloadPercent : replayState === "loading" ? 0 : 100}%`,
                    }}
                  />
                </div>
              </div>

              <div
                className={`step ${replayState === "exporting" ? "active" : ""}`}
              >
                <div className="step-line">
                  <span>Build replay</span>
                  <span className="mono">
                    {replayState === "exporting"
                      ? exportPercent === null
                        ? "working"
                        : `${exportPercent}%`
                      : ""}
                  </span>
                </div>
                <div
                  className={`bar ${replayState === "exporting" && exportPercent === null ? "indeterminate" : ""}`}
                >
                  <div style={{ width: `${exportPercent ?? 0}%` }} />
                </div>
              </div>

              {replayState === "failed" ? (
                <p className="form-error">{historyStatusInfo.error}</p>
              ) : (
                <p className="hint">
                  The skip buttons unlock once the replay starts playing.
                </p>
              )}
            </section>
          )}

          <section className="panel-section">
            <h2>Airspace</h2>
            <div className="stat">
              <span className="stat-value mono">
                {aircraftList.length.toLocaleString()}
              </span>
              <span className="stat-label">
                aircraft {historyMode ? "in replay" : "tracked"}
              </span>
            </div>
            <div className={`status-line ${emergencyCount > 0 ? "alert" : ""}`}>
              <span className={`dot ${emergencyCount > 0 ? "red" : "green"}`} />
              {emergencyCount > 0
                ? `${emergencyCount} aircraft squawking an emergency`
                : "No emergency squawks"}
            </div>
            {emergencyCount > 0 && (
              <ul className="emergency-list">
                {emergencyAircraft.map((a) => (
                  <li key={a.hex}>
                    <button
                      onClick={() => followAircraft(a, false)}
                      title="Zoom to this aircraft and follow it"
                    >
                      <span className="mono">{a.flight?.trim() || a.hex}</span>
                      <span className="emergency-detail mono">
                        {a.squawk ?? "----"} · {a.emergency}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="panel-section anomaly-section">
            <div className="section-head anomaly-heading">
              <h2>Detected anomalies</h2>
              <span className="anomaly-count mono">
                {detectedAnomalies.length}
              </span>
            </div>
            <div className="anomaly-legend" aria-label="Aircraft alert colors">
              <span><i className="dot orange" /> Detector</span>
              <span><i className="dot red" /> Emergency squawk</span>
            </div>
            {detectedAnomalies.length === 0 ? (
              <p className="anomaly-empty">No active anomalies detected</p>
            ) : (
              <ul className="anomaly-list" aria-live="polite">
                {detectedAnomalies.map(({ aircraft, anomaly, riskScore }) => {
                  const isSquawk = anomaly.type === "EMERGENCY_SQUAWK";
                  const callsign = aircraft.flight?.trim() || aircraft.hex;
                  return (
                    <li key={`${aircraft.hex}-${anomaly.type}`}>
                      <button
                        className={`anomaly-card ${isSquawk ? "squawk" : "detected"}`}
                        onClick={() => selectAircraft(aircraft)}
                        title="Select and centre this aircraft"
                      >
                        <span className="anomaly-card-heading">
                          <strong>{callsign}</strong>
                          <span className="anomaly-source">
                            {isSquawk ? "SQUAWK" : "DETECTOR"}
                          </span>
                        </span>
                        <span className="anomaly-card-meta">
                          <span className="anomaly-type">
                            {anomaly.type.toLowerCase().replaceAll("_", " ")}
                          </span>
                        </span>
                        <span className="anomaly-message">
                          {anomaly.message}
                        </span>
                        <span className="anomaly-card-footer mono">
                          <span>{aircraft.hex}</span>
                          <span>Risk {Math.round(riskScore * 100)}%</span>
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>

          <section className="panel-section">
            <h2>Feed</h2>
            <dl className="kv">
              <dt>Source</dt>
              <dd>{historyMode ? "adsb.lol archive" : "adsb.lol live"}</dd>
              <dt>Status</dt>
              <dd className={isLive ? "ok" : "bad"}>
                {isLive ? "OK" : "Offline"}
              </dd>
              <dt>Last update</dt>
              <dd>{lastUpdated}</dd>
            </dl>
          </section>
        </aside>
      </div>
    </div>
  );
}

export default App;
