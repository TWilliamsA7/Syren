import React, { useState } from 'react';
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

function App() {
  // Raw JSON sample dataset representing live/historical US feeds
  const mockApiResponse = {
    "ac": [
      {
        "hex": "ac0094",
        "type": "adsb_icao",
        "flight": "SWA4400 ",
        "r": "N8723Q",
        "t": "B38M",
        "alt_baro": 3275,
        "alt_geom": 3400,
        "gs": 208.8,
        "track": 97.43,
        "baro_rate": -1152,
        "squawk": "3263",
        "emergency": "none",
        "category": "A3",
        "nav_qnh": 1016.0,
        "nav_altitude_mcp": 96,
        "nav_heading": 187.73,
        "lat": 27.812355,
        "lon": -82.584862,
        "messages": 220550,
        "seen": 0.3,
        "rssi": -9.8,
        "dst": 76.910,
        "dir": 241.5
      },
      {
        "hex": "a23b55",
        "type": "adsb_icao",
        "flight": "MXY1606 ",
        "r": "N243BZ",
        "t": "BCS3",
        "alt_baro": "ground",
        "gs": 0.0,
        "squawk": "4041",
        "emergency": "none",
        "category": "A3",
        "lat": 27.975055,
        "lon": -82.532730,
        "messages": 8191,
        "seen": 4.2,
        "rssi": -23.8,
        "dst": 70.106,
        "dir": 247.4
      },
      {
        "hex": "a7e93b",
        "type": "adsb_icao",
        "flight": "FFT2280 ",
        "r": "N609FR",
        "t": "A21N",
        "alt_baro": 17675,
        "alt_geom": 18600,
        "gs": 408.4,
        "track": 142.26,
        "baro_rate": -2240,
        "squawk": "5714",
        "emergency": "none",
        "category": "A3",
        "messages": 80024,
        "seen": 0.2,
        "rssi": -21.3,
        "dst": 49.257,
        "dir": 320.7
      }
    ],
    "msg": "No error",
    "now": 1790400021000,
    "total": 19420 // Simulated grand total across US Airspace for date
  };

  const [aircraftList] = useState<Aircraft[]>(mockApiResponse.ac);
  const [currentRegion, setCurrentRegion] = useState<'US_ALL' | 'US_WEST' | 'US_CENTRAL' | 'US_EAST'>('US_ALL');
  const [simState, setSimState] = useState<'running' | 'paused'>('running');

  // Search state (Date is required, ICAO is optional)
  const [searchDate, setSearchDate] = useState<string>('2026-09-26');
  const [searchIcao, setSearchIcao] = useState<string>('');
  
  // View states: 'live' | 'search_result'
  const [viewMode, setViewMode] = useState<'live' | 'search_result'>('live');
  const [searchedAircraft, setSearchedAircraft] = useState<Aircraft | null>(null);
  const [searchError, setSearchError] = useState<string>('');

  // Handle Search Submission (Date is required)
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setSearchError('');

    if (!searchDate) {
      setSearchError('A date is required to query the archive.');
      return;
    }

    const query = searchIcao.trim().toLowerCase();

    // If ICAO is provided, look for that specific plane
    if (query) {
      const found = aircraftList.find(
        a => 
          a.hex.toLowerCase() === query || 
          (a.r && a.r.toLowerCase() === query) || 
          (a.flight && a.flight.trim().toLowerCase() === query)
      );

      if (found) {
        setSearchedAircraft({ ...found, timestamp: searchDate });
      } else {
        // Fallback generator for demonstration if searching outside current sample slice
        setSearchedAircraft({
          hex: query.toUpperCase(),
          type: 'adsb_icao',
          flight: query.toUpperCase(),
          r: 'N-CUSTOM',
          t: 'B738',
          alt_baro: 36000,
          gs: 465.2,
          track: 180.0,
          baro_rate: 0,
          squawk: '1200',
          emergency: 'none',
          category: 'A3',
          messages: 45100,
          rssi: -14.2,
          lat: 34.05,
          lon: -118.24,
          timestamp: searchDate
        });
      }
    } else {
      // Date only: Show aggregate daily feed stats
      setSearchedAircraft(null);
    }

    setViewMode('search_result');
  };

  const returnToLive = () => {
    setViewMode('live');
    setSearchedAircraft(null);
    setSearchIcao('');
    setSearchError('');
  };

  const utcTimestamp = new Date(mockApiResponse.now).toUTCString();
  const hasFeedAnomaly = aircraftList.some(a => a.emergency && a.emergency !== 'none');

  return (
    <div className="syren-container">
      
      {/* Top Header */}
      <header className="syren-header">
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '1rem' }}>
          <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '800', letterSpacing: '-0.025em' }}>
            SYREN <span style={{ color: 'var(--accent)', fontSize: '0.875rem', fontWeight: '400' }}>// Edge Aviation Crisis Engine</span>
          </h1>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', borderLeft: '1px solid var(--border)', paddingLeft: '1rem' }}>
            {viewMode === 'live' 
              ? '🟢 ADS-B Live US Airspace Feed' 
              : searchedAircraft ? `🔍 Historical Archive: ICAO [${searchedAircraft.hex}] (${searchDate})` : `📅 US Airspace Aggregate (${searchDate})`}
          </span>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          {/* Always accessible live toggle button */}
          <button 
            onClick={returnToLive}
            style={{ 
              backgroundColor: viewMode === 'live' ? 'rgba(54, 246, 180, 0.15)' : 'var(--accent)', 
              color: viewMode === 'live' ? 'var(--green)' : 'var(--text-main)', 
              border: viewMode === 'live' ? '1px solid var(--green)' : 'none', 
              padding: '0.375rem 0.75rem', 
              borderRadius: '0.375rem', 
              fontWeight: '700', 
              cursor: 'pointer', 
              fontSize: '0.75rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem'
            }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: viewMode === 'live' ? 'var(--green)' : '#0F172A' }}></span>
            {viewMode === 'live' ? 'Live Feed Active' : 'Return to Live'}
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
        
        {/* Left Side: Map with Regional Zoom */}
        <div className="map-placeholder">
          <div className="map-grid-bg" />

          {/* Region Zoom Toolbar */}
          <div className="map-toolbar">
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', paddingRight: '0.25rem' }}>Region Zoom:</span>
            <button className={currentRegion === 'US_ALL' ? 'active' : ''} onClick={() => { setCurrentRegion('US_ALL'); if(viewMode !== 'live') returnToLive(); }}>Continental US</button>
            <button className={currentRegion === 'US_WEST' ? 'active' : ''} onClick={() => { setCurrentRegion('US_WEST'); if(viewMode !== 'live') returnToLive(); }}>West Coast</button>
            <button className={currentRegion === 'US_CENTRAL' ? 'active' : ''} onClick={() => { setCurrentRegion('US_CENTRAL'); if(viewMode !== 'live') returnToLive(); }}>Central US</button>
            <button className={currentRegion === 'US_EAST' ? 'active' : ''} onClick={() => { setCurrentRegion('US_EAST'); if(viewMode !== 'live') returnToLive(); }}>East Coast</button>
          </div>

          {/* Map Center Placeholder */}
          <div style={{ zIndex: 10, textAlign: 'center', padding: '2rem' }}>
            <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--accent)', marginBottom: '0.5rem' }}>
              {viewMode === 'live' 
                ? `ADS-B Live US Airspace (${currentRegion.replace('_', ' ')})` 
                : searchedAircraft 
                  ? `Target Tracking: ${searchedAircraft.flight?.trim() || searchedAircraft.hex} (${searchDate})` 
                  : `US National Airspace Archive (${searchDate})`}
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', maxWidth: '450px', margin: '0 auto 1.5rem auto' }}>
              {viewMode === 'live' 
                ? 'Streaming real-time normalized US flight vectors over WebSocket /ws/state.' 
                : searchedAircraft 
                  ? `Locked onto ICAO Hex [${searchedAircraft.hex}] at Lat: ${searchedAircraft.lat}, Lon: ${searchedAircraft.lon}`
                  : `Displaying aggregate daily traffic metrics and anomaly statuses for ${searchDate}.`}
            </p>
            <div style={{ display: 'inline-flex', gap: '1.5rem', backgroundColor: 'var(--bg-sidebar)', padding: '0.75rem 1.25rem', borderRadius: '0.5rem', border: '1px solid var(--border)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8125rem' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--green)' }}></span> 
                {searchedAircraft 
                  ? `Status: ${searchedAircraft.emergency === 'none' ? 'Nominal' : 'EMERGENCY'}` 
                  : `Total Active Flights (${searchDate}): ${mockApiResponse.total.toLocaleString()}`}
              </span>
            </div>
          </div>
        </div>

        {/* Right Side: Sidebar (Search & Feed Information) */}
        <div className="sidebar">
          
          {/* Historical Search Box (Date Required, ICAO Optional) */}
          <div className="search-box">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
              <div style={{ fontSize: '0.8125rem', fontWeight: '700', color: 'var(--accent)' }}>
                🔍 Date & Optional ICAO Search
              </div>
              {viewMode === 'search_result' && (
                <button 
                  onClick={returnToLive}
                  style={{ background: 'none', border: 'none', color: 'var(--green)', fontSize: '0.7rem', cursor: 'pointer', textDecoration: 'underline', padding: 0 }}>
                  Exit Search
                </button>
              )}
            </div>
            <form onSubmit={handleSearch}>
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <div style={{ flex: 1 }}>
                  <label style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Date (Required)</label>
                  <input 
                    type="date" 
                    value={searchDate} 
                    onChange={(e) => setSearchDate(e.target.value)} 
                    style={{ margin: '0.25rem 0 0 0' }}
                  />
                </div>
                <div style={{ flex: 1.2 }}>
                  <label style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>ICAO / Callsign (Optional)</label>
                  <input 
                    type="text" 
                    placeholder="e.g. ac0094 or SWA4400" 
                    value={searchIcao} 
                    onChange={(e) => setSearchIcao(e.target.value)} 
                    style={{ margin: '0.25rem 0 0 0' }}
                  />
                </div>
              </div>
              {searchError && <div style={{ color: 'var(--red)', fontSize: '0.7rem', marginBottom: '0.375rem' }}>{searchError}</div>}
              <button 
                type="submit" 
                style={{ width: '100%', backgroundColor: 'var(--accent)', color: 'var(--text-main)', border: 'none', padding: '0.4rem', borderRadius: '0.375rem', fontWeight: 'bold', fontSize: '0.75rem', cursor: 'pointer' }}>
                Run Search
              </button>
            </form>
          </div>

          {/* VIEW MODE 1: LIVE FEED OR DATE SEARCH WITHOUT ICAO (Shows Total Flights + Anomaly Status + Metadata) */}
          {(!searchedAircraft) && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', flex: 1 }}>
              
              {/* Total Flights Card */}
              <div style={{ backgroundColor: 'var(--bg-sidebar)', border: '1px solid var(--border)', borderRadius: '0.5rem', padding: '1.25rem' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>
                  {viewMode === 'live' ? 'Current US Airspace Volume' : `Recorded Volume (${searchDate})`}
                </div>
                <div style={{ fontSize: '2rem', fontWeight: '800', color: 'var(--accent)' }}>{mockApiResponse.total.toLocaleString()}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Active transponders reporting via ADS-B</div>
              </div>

              {/* Feed Anomaly Status Card */}
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
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0.5rem 0 0 0' }}>
                  Continuous boundary evaluations running on local Jetson node.
                </p>
              </div>

              {/* Timestamps & Metadata Card */}
              <div style={{ backgroundColor: 'var(--bg-sidebar)', border: '1px solid var(--border)', borderRadius: '0.5rem', padding: '1.25rem', fontSize: '0.8125rem' }}>
                <div style={{ fontWeight: '700', color: 'var(--text-main)', marginBottom: '0.75rem' }}>Telemetry System Metadata</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>
                  <span>Query Timestamp:</span>
                  <strong style={{ color: 'var(--text-main)' }}>{viewMode === 'live' ? utcTimestamp : `${searchDate} 00:00:00 UTC`}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>
                  <span>API Status:</span>
                  <strong style={{ color: 'var(--green)' }}>{mockApiResponse.msg}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                  <span>Sample Slice Size:</span>
                  <strong style={{ color: 'var(--text-main)' }}>{aircraftList.length} local packets</strong>
                </div>
              </div>

              <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '1rem' }}>
                💡 Tip: To inspect a specific plane, enter a date and type an ICAO hex or callsign (e.g., <code style={{ color: 'var(--accent)' }}>ac0094</code>).
              </div>

            </div>
          )}

          {/* VIEW MODE 2: DATE + ICAO SEARCHED (Shows Deep Dive Single Aircraft Telemetry & Removes Total Flights) */}
          {searchedAircraft && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', flex: 1, overflowY: 'auto' }}>
              
              <div style={{ 
                backgroundColor: 'var(--bg-sidebar)', 
                border: `1px solid ${searchedAircraft.emergency !== 'none' ? 'var(--red)' : 'var(--accent)'}`, 
                borderRadius: '0.5rem', 
                padding: '1.25rem' 
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <div>
                    <h3 style={{ margin: 0, fontSize: '1.25rem', fontWeight: '800' }}>{searchedAircraft.flight?.trim() || 'Unknown Flight'}</h3>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ICAO Hex: <strong>{searchedAircraft.hex}</strong> | Reg: <strong>{searchedAircraft.r || 'N/A'}</strong></span>
                  </div>
                  <span style={{ 
                    backgroundColor: searchedAircraft.emergency !== 'none' ? 'var(--red)' : 'var(--green)', 
                    color: searchedAircraft.emergency !== 'none' ? 'var(--text-main)' : '#0F172A', 
                    fontSize: '0.625rem', 
                    fontWeight: '700', 
                    padding: '0.25rem 0.625rem', 
                    borderRadius: '9999px',
                    textTransform: 'uppercase'
                  }}>
                    {searchedAircraft.emergency !== 'none' ? `Emergency: ${searchedAircraft.emergency}` : 'Nominal'}
                  </span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.8125rem', color: '#CBD5E1', marginTop: '1rem', borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
                  <div>Aircraft Type: <strong style={{ color: 'var(--text-main)' }}>{searchedAircraft.t || 'N/A'}</strong></div>
                  <div>Category: <strong style={{ color: 'var(--text-main)' }}>{searchedAircraft.category || 'N/A'}</strong></div>
                  <div>Baro Altitude: <strong style={{ color: 'var(--text-main)' }}>{searchedAircraft.alt_baro} ft</strong></div>
                  <div>Geometric Alt: <strong style={{ color: 'var(--text-main)' }}>{searchedAircraft.alt_geom ? `${searchedAircraft.alt_geom} ft` : 'N/A'}</strong></div>
                  <div>Ground Speed: <strong style={{ color: 'var(--text-main)' }}>{searchedAircraft.gs} kts</strong></div>
                  <div>Track / Heading: <strong style={{ color: 'var(--text-main)' }}>{searchedAircraft.track ?? searchedAircraft.nav_heading ?? 'N/A'}°</strong></div>
                  <div>Baro Rate: <strong style={{ color: 'var(--text-main)' }}>{searchedAircraft.baro_rate ?? 0} fpm</strong></div>
                  <div>Squawk Code: <strong style={{ color: 'var(--green)' }}>{searchedAircraft.squawk || 'N/A'}</strong></div>
                  <div>Signal RSSI: <strong style={{ color: 'var(--text-main)' }}>{searchedAircraft.rssi ?? 'N/A'} dB</strong></div>
                  <div>Total Packets: <strong style={{ color: 'var(--text-main)' }}>{searchedAircraft.messages?.toLocaleString() || 'N/A'}</strong></div>
                </div>

                {searchedAircraft.nav_qnh && (
                  <div style={{ marginTop: '1rem', backgroundColor: '#0F172A', padding: '0.75rem', borderRadius: '0.375rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    <div>Nav QNH: <strong style={{ color: 'var(--text-main)' }}>{searchedAircraft.nav_qnh} hPa</strong></div>
                    <div>Nav MCP Altitude: <strong style={{ color: 'var(--text-main)' }}>{searchedAircraft.nav_altitude_mcp} meters</strong></div>
                  </div>
                )}

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem', borderTop: '1px solid var(--border)', paddingTop: '0.75rem' }}>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Archived Date: <strong>{searchDate}</strong></span>
                  <button 
                    onClick={returnToLive}
                    style={{ backgroundColor: 'transparent', border: '1px solid var(--accent)', color: 'var(--accent)', fontSize: '0.7rem', fontWeight: 'bold', padding: '0.25rem 0.5rem', borderRadius: '0.25rem', cursor: 'pointer' }}>
                    Return to Live
                  </button>
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