import React, { useState, useEffect, useRef } from 'react';
import DeckGL from '@deck.gl/react';
import Map from 'react-map-gl/maplibre'; // or 'react-map-gl' for Mapbox
import { IconLayer } from '@deck.gl/layers';

import './App.css';

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
}

const DEFAULT_VIEW_STATE = {
  longitude: -95.7129,
  latitude: 37.0902,
  zoom: 4,
  maxZoom: 18,
  minZoom: 2,
  pitch: 0,
  bearing: 0
};

// Region specific view states for zooming buttons
const REGION_VIEWS = {
  US_ALL: { longitude: -95.7129, latitude: 37.0902, zoom: 4, pitch: 0, bearing: 0 },
  US_WEST: { longitude: -120.5583, latitude: 40.5556, zoom: 4.7, pitch: 0, bearing: 0 },
  US_MIDWEST: { longitude: -101.6298, latitude: 41.8781, zoom: 5, pitch: 0, bearing: 0 },
  US_SOUTH: { longitude: -93.7970, latitude: 31.7767, zoom: 5.15, pitch: 0, bearing: 0 },
  US_EAST: { longitude: -75.1652, latitude: 39.9526, zoom: 5, pitch: 0, bearing: 0 }
};

// Inline SVG Atlas for the plane icon with black outline/stroke
const AIRPLANE_ICON = 'data:image/svg+xml;charset=utf-8,<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 24 24" fill="%2336F6B4" stroke="black" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"><path d="M12 2a1.5 1.5 0 0 1 1.5 1.5v5.25l7 3.75v1.75l-7-2.25v5l2 1.5v1.25l-3.5-1-3.5 1v-1.25l2-1.5v-5l-7 2.25v-1.75l7-3.75V3.5A1.5 1.5 0 0 1 12 2z"/></svg>';
const ICON_MAPPING = {
  marker: { x: 0, y: 0, width: 128, height: 128, mask: false }
};

function App() {
  const [aircraftList, setAircraftList] = useState<Aircraft[]>([]);
  const [isLive, setIsLive] = useState<boolean>(true);
  const [lastUpdated, setLastUpdated] = useState<string>('Initializing...');
  
  const [currentRegion, setCurrentRegion] = useState<'US_ALL' | 'US_WEST' | 'US_MIDWEST' | 'US_SOUTH' | 'US_EAST'>('US_ALL');
  const [viewState, setViewState] = useState(DEFAULT_VIEW_STATE);
  const [simState, setSimState] = useState<'running' | 'paused'>('running');

  // Search state
  const [searchDate, setSearchDate] = useState<string>('2026-09-24');
  const [searchStartTime, setSearchStartTime] = useState<string>('16:00');
  const [searchIcao, setSearchIcao] = useState<string>('');
  
  // History / Replay state
  const [historyMode, setHistoryMode] = useState<boolean>(false);
  const [historyStatusInfo, setHistoryStatusInfo] = useState<any>({ state: 'idle' });
  const [replayClockFormatted, setReplayClockFormatted] = useState<string>('');

  // View states
  const [viewMode, setViewMode] = useState<'live' | 'search_result'>('live');
  const [searchedAircraft, setSearchedAircraft] = useState<Aircraft | null>(null);
  const [searchError, setSearchError] = useState<string>('');
  const [hoveredAircraft, setHoveredAircraft] = useState<Aircraft | null>(null);

  const consecutiveFailuresRef = useRef<number>(0);

  // Helper function to safely parse either JSON array or NDJSON (JSON Lines)
  const parseJsonData = (text: string) => {
    try {
      return JSON.parse(text);
    } catch {
      // Fallback for NDJSON / JSON Lines format
      return text
        .trim()
        .split('\n')
        .filter(line => line.trim().length > 0)
        .map(line => JSON.parse(line));
    }
  };

  // Poll history_status if history mode is active
  useEffect(() => {
    let statusTimer: NodeJS.Timeout;
    let cancelled = false;  // set on cleanup, so a fetch still in flight can't keep an old loop going
    if (historyMode) {
      const pollStatus = async () => {
        try {
          const res = await fetch('/api/history_status', { cache: 'no-store' });
          if (res.ok) {
            const data = await res.json();
            if (cancelled) return;
            setHistoryStatusInfo(data);
            if (data.clock) {
              const dateObj = new Date(data.clock * 1000);
              setReplayClockFormatted(dateObj.toUTCString());
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

  // Poll data file continuously (/history.jsonl if historyMode, else /data.jsonl)
  useEffect(() => {
    let timer: NodeJS.Timeout;
    let cancelled = false;  // set on cleanup, so a fetch still in flight can't keep an old loop going

    const fetchData = async () => {
      try {
        const endpoint = historyMode ? '/history.jsonl' : '/data.jsonl';
        const response = await fetch(endpoint, { cache: 'no-store' });
        if (!response.ok) throw new Error('Failed to fetch data file');

        const rawText = await response.text();
        if (cancelled) return;
        const rawApiResponse = parseJsonData(rawText);

        const parsedData: Aircraft[] = rawApiResponse.map((item: any) => ({
          hex: item.icao24,
          flight: item.flight_id,
          type: item.aircraft?.type_code || 'adsb_icao',
          t: item.aircraft?.type_code || undefined,
          r: item.aircraft?.registration || undefined,
          lat: item.position?.latitude,
          lon: item.position?.longitude,
          alt_baro: item.position?.altitude_baro_ft,
          alt_geom: item.position?.altitude_geom_ft,
          gs: item.kinematics?.ground_speed_kts,
          track: item.kinematics?.track_deg,
          baro_rate: item.kinematics?.vertical_rate_baro_fpm,
          squawk: item.status?.squawk,
          emergency: item.status?.emergency,
          category: item.aircraft?.category,
          timestamp: item.timestamp ? new Date(item.timestamp * 1000).toISOString() : new Date().toISOString()
        }));

        setAircraftList(parsedData);
        setIsLive(true);  // COOKED only when the fetch fails, not during every replay
        setLastUpdated(new Date().toUTCString());
        consecutiveFailuresRef.current = 0;

        // Schedule next standard check interval (1 second)
        timer = setTimeout(fetchData, 1000);
      } catch (err) {
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
    try {
      const response = await fetch('/api/start_history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date, start: startTime })
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
      await fetch('/api/stop_history', { method: 'POST' });
      setHistoryMode(false);
      setHistoryStatusInfo({ state: 'idle' });
    } catch (e) {
      console.error("Error calling stop_history", e);
    }
  };

  const skipTime = async (seconds: number) => {
    try {
      await fetch('/api/skip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seconds })
      });
    } catch (e) {
      console.error("Error calling skip", e);
    }
  };

  // Handle Search Submission
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setSearchError('');

    if (!searchDate) {
      setSearchError('A date is required to query the archive.');
      return;
    }

    // Trigger backend history replay session for the selected date and start time
    startHistory(searchDate, searchStartTime);

    const query = searchIcao.trim().toLowerCase();
    if (query) {
      const found = aircraftList.find(
        a => 
          a.hex.toLowerCase() === query || 
          (a.r && a.r.toLowerCase() === query) || 
          (a.flight && a.flight.trim().toLowerCase() === query)
      );

      if (found) {
        setSearchedAircraft({ ...found, timestamp: searchDate });
        if (found.lat && found.lon) {
          setViewState(v => ({
            ...v,
            longitude: found.lon!,
            latitude: found.lat!,
            zoom: 9
          }));
        }
      }
    } else {
      setSearchedAircraft(null);
    }

    setViewMode('search_result');
  };

  const returnToLive = () => {
    stopHistory();
    setViewMode('live');
    setSearchedAircraft(null);
    setSearchIcao('');
    setSearchError('');
  };

  const handleRegionChange = (region: 'US_ALL' | 'US_WEST' | 'US_MIDWEST' | 'US_SOUTH' | 'US_EAST') => {
    setCurrentRegion(region);
    setViewState(v => ({
      ...v,
      ...REGION_VIEWS[region]
    }));
  };

  const hasFeedAnomaly = aircraftList.some(a => a.emergency && a.emergency !== 'none');

  // Define Deck.GL Layers for Aircraft visualization mapped to geo coordinates
  const layers = [
    new IconLayer({
      id: 'aircraft-icon-layer',
      data: viewMode === 'live' 
        ? aircraftList 
        : (searchedAircraft ? [searchedAircraft] : aircraftList),
      iconAtlas: AIRPLANE_ICON,
      iconMapping: ICON_MAPPING,
      getIcon: () => 'marker',
      getPosition: (d: Aircraft) => [d.lon ?? -95.7129, d.lat ?? 37.0902],
      getSize: 24,
      getAngle: (d: any) => {
        const rawHeading = d.true_heading ?? d.nav_heading ?? d.heading ?? d.track ?? 0;
        return -rawHeading; 
      },
      getColor: (d: Aircraft) => 
        (d.emergency && d.emergency !== 'none') ? [244, 63, 94] : [54, 246, 180],
      pickable: true,
      onHover: info => setHoveredAircraft(info.object as Aircraft || null),
      updateTriggers: {
        data: [aircraftList, searchedAircraft, viewMode],
        getAngle: [aircraftList, searchedAircraft, viewMode] 
      }
    })
  ];

  return (
    <div className="syren-container">
      
      {/* Top Header */}
      <header className="syren-header">
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '1rem' }}>
          <h1 style={{ margin: 0, color: 'var(--text-main)', fontSize: '1.5rem', fontWeight: '800', letterSpacing: '-0.025em' }}>
            SYREN <span style={{ color: 'var(--accent)', fontSize: '0.875rem', fontWeight: '400' }}>// Edge Aviation Crisis Engine</span>
          </h1>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', borderLeft: '1px solid var(--border)', paddingLeft: '1rem' }}>
            {!historyMode 
              ? 'ADS-B Live US Airspace Feed' 
              : `Replay Mode: ${searchDate} @ ${searchStartTime} | ${replayClockFormatted || historyStatusInfo.state}`}
          </span>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <button 
            onClick={returnToLive}
            style={{ 
              backgroundColor: isLive ? 'rgba(54, 246, 180, 0.15)' : 'rgba(244, 63, 94, 0.15)', 
              color: isLive ? 'var(--green)' : 'var(--red)', 
              border: `1px solid ${isLive ? 'var(--green)' : 'var(--red)'}`, 
              padding: '0.375rem 0.75rem', 
              borderRadius: '0.375rem', 
              fontWeight: '700', 
              cursor: 'pointer', 
              fontSize: '0.75rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem'
            }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: isLive ? 'var(--green)' : 'var(--red)' }}></span>
            {isLive ? 'Live Feed Active' : "IT'S COOKED - NOT LIVE ANYMORE"}
          </button>

          <button 
            onClick={() => setSimState(simState === 'running' ? 'paused' : 'running')}
            style={{ backgroundColor: simState === 'running' ? 'var(--red)' : 'var(--green)', color: '#0F172A', border: 'none', padding: '0.375rem 0.75rem', borderRadius: '0.375rem', fontWeight: '700', cursor: 'pointer', fontSize: '0.75rem' }}>
            {simState === 'running' ? 'POST /api/pause' : 'POST /api/start'}
          </button>
        </div>
      </header>

      {/* Workspace Grid */}
      <div className="syren-workspace">
        
        {/* Left Side: Coordinate-mapped Deck.GL Map Canvas & Earth Background */}
        <div className="map-placeholder" style={{ position: 'relative', overflow: 'hidden' }}>
          <div className="map-grid-bg" style={{ zIndex: 1, pointerEvents: 'none' }} />

          {/* Deck.GL Canvas Integration with Geographic Coordinate mapping */}
          <DeckGL
            viewState={viewState}
            onViewStateChange={(e: any) => setViewState(e.viewState)}
            controller={true}
            layers={layers}
            style={{ position: 'absolute', inset: 0, zIndex: 2 }}
          >
            <Map
              reuseMaps
              mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
            />
          </DeckGL>

          {/* Hover Tooltip Overlay */}
          {hoveredAircraft && (
            <div style={{
              position: 'absolute',
              bottom: '2rem',
              left: '2rem',
              backgroundColor: 'rgba(15, 23, 42, 0.9)',
              border: '1px solid var(--accent)',
              padding: '0.75rem 1rem',
              borderRadius: '0.5rem',
              zIndex: 30,
              fontSize: '0.75rem',
              color: 'var(--text-main)',
              pointerEvents: 'none',
              backdropFilter: 'blur(4px)'
            }}>
              <div>Flight: <strong style={{ color: 'var(--accent)' }}>{hoveredAircraft.flight?.trim() || 'N/A'}</strong> ({hoveredAircraft.hex})</div>
              <div>Alt: <strong>{hoveredAircraft.alt_baro} ft</strong> | GS: <strong>{hoveredAircraft.gs} kts</strong></div>
              <div>Squawk: <strong style={{ color: 'var(--green)' }}>{hoveredAircraft.squawk || 'N/A'}</strong></div>
            </div>
          )}

          {/* Region Zoom Toolbar */}
          <div className="map-toolbar" style={{ zIndex: 20 }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', paddingRight: '0.25rem' }}>Region Zoom:</span>
            <button className={currentRegion === 'US_ALL' ? 'active' : ''} onClick={() => handleRegionChange('US_ALL')}>Continental US</button>
            <button className={currentRegion === 'US_WEST' ? 'active' : ''} onClick={() => handleRegionChange('US_WEST')}>West Coast</button>
            <button className={currentRegion === 'US_MIDWEST' ? 'active' : ''} onClick={() => handleRegionChange('US_MIDWEST')}>Midwest US</button>
            <button className={currentRegion === 'US_SOUTH' ? 'active' : ''} onClick={() => handleRegionChange('US_SOUTH')}>South US</button>
            <button className={currentRegion === 'US_EAST' ? 'active' : ''} onClick={() => handleRegionChange('US_EAST')}>East Coast</button>
          </div>

          {/* Map Status Badge */}
          <div style={{ position: 'absolute', bottom: '1.5rem', right: '1.5rem', zIndex: 20, backgroundColor: 'rgba(15, 23, 42, 0.85)', padding: '0.5rem 1rem', borderRadius: '0.375rem', border: '1px solid var(--border)', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            Deck.GL Active Layer Rendering ({aircraftList.length} entities)
          </div>
        </div>

        {/* Right Side: Sidebar (Search & Feed Information) */}
        <div className="sidebar">
          
          {/* Historical Search Box & Replay Control Buttons */}
          <div className="search-box">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
              <div style={{ fontSize: '0.8125rem', fontWeight: '700', color: 'var(--accent)' }}>
                History Replay & Archive Controls
              </div>
              {historyMode && (
                <button 
                  onClick={returnToLive}
                  style={{ background: 'none', border: 'none', color: 'var(--green)', fontSize: '0.7rem', cursor: 'pointer', textDecoration: 'underline', padding: 0 }}>
                  Exit Date Mode
                </button>
              )}
            </div>
            
            <form onSubmit={handleSearch}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <div>
                  <label style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Date (Required)</label>
                  <input 
                    type="date" 
                    value={searchDate} 
                    onChange={(e) => setSearchDate(e.target.value)} 
                    style={{ margin: '0.25rem 0 0 0', width: '100%' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Start Time (UTC)</label>
                  <input 
                    type="time" 
                    value={searchStartTime} 
                    onChange={(e) => setSearchStartTime(e.target.value)} 
                    style={{ margin: '0.25rem 0 0 0', width: '100%' }}
                  />
                </div>
              </div>

              <div style={{ marginBottom: '0.5rem' }}>
                <label style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>ICAO / Callsign</label>
                <input 
                  type="text" 
                  placeholder="e.g. ac0094" 
                  value={searchIcao} 
                  onChange={(e) => setSearchIcao(e.target.value)} 
                  style={{ margin: '0.25rem 0 0 0', width: '100%' }}
                />
              </div>

              {searchError && <div style={{ color: 'var(--red)', fontSize: '0.7rem', marginBottom: '0.375rem' }}>{searchError}</div>}
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.35rem', marginBottom: '0.35rem' }}>
                <button 
                  type="button" 
                  onClick={() => startHistory(searchDate, searchStartTime)}
                  style={{ backgroundColor: 'var(--accent)', color: 'var(--text-main)', border: 'none', padding: '0.4rem', borderRadius: '0.375rem', fontWeight: 'bold', fontSize: '0.7rem', cursor: 'pointer' }}>
                  Enter a date
                </button>
                <button 
                  type="button" 
                  onClick={returnToLive}
                  style={{ backgroundColor: 'rgba(244, 63, 94, 0.2)', color: 'var(--red)', border: '1px solid var(--red)', padding: '0.4rem', borderRadius: '0.375rem', fontWeight: 'bold', fontSize: '0.7rem', cursor: 'pointer' }}>
                  Exit date mode
                </button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.35rem' }}>
                <button 
                  type="button" 
                  onClick={() => skipTime(-300)}
                  disabled={!historyMode}
                  style={{ backgroundColor: historyMode ? '#334155' : '#1e293b', color: historyMode ? '#f8fafc' : '#64748b', border: '1px solid var(--border)', padding: '0.35rem', borderRadius: '0.375rem', fontSize: '0.7rem', cursor: historyMode ? 'pointer' : 'not-allowed' }}>
                  &lt;&lt; Go back (-5m)
                </button>
                <button 
                  type="button" 
                  onClick={() => skipTime(300)}
                  disabled={!historyMode}
                  style={{ backgroundColor: historyMode ? '#334155' : '#1e293b', color: historyMode ? '#f8fafc' : '#64748b', border: '1px solid var(--border)', padding: '0.35rem', borderRadius: '0.375rem', fontSize: '0.7rem', cursor: historyMode ? 'pointer' : 'not-allowed' }}>
                  Go forward (+5m) &gt;&gt;
                </button>
              </div>
            </form>
          </div>

          {/* REPLAY STATUS PANEL */}
          {historyMode && (
            <div style={{ backgroundColor: 'var(--bg-sidebar)', border: '1px solid var(--accent)', borderRadius: '0.5rem', padding: '1.25rem' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--accent)', fontWeight: '700', textTransform: 'uppercase', marginBottom: '0.25rem' }}>Replay Status</div>
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-main)', marginBottom: '0.5rem' }}>
                State: <strong style={{ textTransform: 'uppercase' }}>{historyStatusInfo.state}</strong>
              </div>
              
              {/* Progress Bar for Downloading State */}
              {historyStatusInfo.state === 'downloading' && (
                <div style={{ marginBottom: '0.75rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                    <span>Downloading Archive...</span>
                    <span>
                      {historyStatusInfo.total_bytes > 0 
                        ? `${Math.round((historyStatusInfo.done_bytes / historyStatusInfo.total_bytes) * 100)}%` 
                        : '0%'}
                    </span>
                  </div>
                  <div style={{ width: '100%', height: '6px', backgroundColor: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
                    <div 
                      style={{ 
                        width: `${historyStatusInfo.total_bytes > 0 ? (historyStatusInfo.done_bytes / historyStatusInfo.total_bytes) * 100 : 0}%`, 
                        height: '100%', 
                        backgroundColor: 'var(--accent)',
                        transition: 'width 0.2s ease-in-out'
                      }} 
                    />
                  </div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    {(historyStatusInfo.done_bytes / 1e9).toFixed(2)} GB / {(historyStatusInfo.total_bytes / 1e9).toFixed(2)} GB
                  </div>
                </div>
              )}

              {historyStatusInfo.state === 'exporting' && (
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                  Exporting day traces to jsonl...
                </div>
              )}

              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                Replay Clock: <strong style={{ color: 'var(--accent)' }}>{replayClockFormatted || 'Syncing...'}</strong>
              </div>
            </div>
          )}

          {/* LIVE FEED INFO PANEL */}
          {!historyMode && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', flex: 1 }}>
              <div style={{ backgroundColor: 'var(--bg-sidebar)', border: '1px solid var(--border)', borderRadius: '0.5rem', padding: '1.25rem' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>
                  Current US Airspace Volume
                </div>
                <div style={{ fontSize: '2rem', fontWeight: '800', color: 'var(--accent)' }}>{aircraftList.length.toLocaleString()}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Active transponders rendering on Deck.GL</div>
              </div>

              <div style={{ 
                backgroundColor: 'var(--bg-sidebar)', 
                border: `1px solid ${hasFeedAnomaly ? 'var(--red)' : 'var(--green)'}`, 
                borderRadius: '0.5rem', 
                padding: '1.25rem' 
              }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>Overall Feed Anomaly Status</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.125rem', fontWeight: '700', color: hasFeedAnomaly ? 'var(--red)' : 'var(--green)' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: hasFeedAnomaly ? 'var(--red)' : 'var(--green)' }}></span>
                  {hasFeedAnomaly ? 'ALERT: Emergency Squawk Detected' : 'All US Telemetry Nominal'}
                </div>
              </div>

              <div style={{ backgroundColor: 'var(--bg-sidebar)', border: '1px solid var(--border)', borderRadius: '0.5rem', padding: '1.25rem', fontSize: '0.8125rem' }}>
                <div style={{ fontWeight: '700', color: 'var(--text-main)', marginBottom: '0.75rem' }}>Telemetry System Metadata</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>
                  <span>Query Timestamp:</span>
                  <strong style={{ color: 'var(--text-main)' }}>{lastUpdated}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                  <span>API Status:</span>
                  <strong style={{ color: isLive ? 'var(--green)' : 'var(--red)' }}>
                    {isLive ? 'No error' : 'COOKED (Offline)'}
                  </strong>
                </div>
              </div>
            </div>
          )}

          {/* SEARCHED AIRCRAFT */}
          {searchedAircraft && viewMode === 'search_result' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', flex: 1, overflowY: 'auto' }}>
              <div style={{ 
                backgroundColor: 'var(--bg-sidebar)', 
                border: '1px solid var(--accent)', 
                borderRadius: '0.5rem', 
                padding: '1.25rem' 
              }}>
                <h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: '800' }}>{searchedAircraft.flight?.trim() || searchedAircraft.hex}</h3>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                  Alt: <strong>{searchedAircraft.alt_baro} ft</strong> | GS: <strong>{searchedAircraft.gs} kts</strong>
                </div>
              </div>
            </div>
          )}

        </div>

      </div>
    </div>
  );
}

export default App;