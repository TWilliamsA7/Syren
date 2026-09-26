{
  "timestamp": 1790400021.0,           // when the position was measured, Unix seconds | live: now/1000 - seen_pos | history: file timestamp + point[0]
  "icao24": "a5d28c",                  // permanent transponder ID, the join key | live: hex | history: file icao
  "flight_id": "UAL3776",              // callsign, spaces stripped, falls back to icao24 | live: flight | history: point[8].flight, reuse last
  "aircraft": {
    "registration": "N47412",          // tail number | live: r | history: file r
    "type_code": "B39M",               // model code, B39M = 737 MAX 9 | live: t | history: file t
    "category": "A3"                   // size class A1 light, A3 large, A4 heavy, A7 rotorcraft | live: category | history: point[8].category, reuse last
  },
  "position": {
    "latitude": 28.848358,             // GPS latitude, degrees | live: lat | history: point[1]
    "longitude": -82.270907,           // GPS longitude, degrees | live: lon | history: point[2]
    "altitude_baro_ft": 14850,         // pressure altitude, ft, what ATC uses | live: alt_baro | history: point[3]
    "altitude_geom_ft": 15575,         // GPS altitude, ft, can read above or below baro | live: alt_geom | history: point[10]
    "on_ground": false,                // true when parked or taxiing | live: alt_baro == "ground" | history: point[3] == "ground"
    "source": "adsb_icao",             // readsb message type, any readsb value allowed: adsb_icao = aircraft's own broadcast, adsr_icao = rebroadcast (no accuracy data), mlat = ground-estimated, tisb_* = rebroadcast radar track | live: type | history: point[9]
    "accuracy_m": 186,                 // position accuracy radius, smaller is better, null when unknown | live: rc (0 means unknown) | history: not stored
    "stale": false                     // true if no position heard for 20+ s before this | live: seen_pos > 20 | history: point[6] & 1
  },
  "kinematics": {
    "ground_speed_kts": 423.2,         // speed over the ground, knots, includes wind | live: gs | history: point[4]
    "track_deg": 174.17,               // direction of travel, degrees, 0 = north | live: track, or true_heading when on the ground | history: point[5]
    "vertical_rate_baro_fpm": -2560,   // climb (+) or descent (-), ft/min, from pressure, preferred | live: baro_rate | history: point[7]
    "vertical_rate_geom_fpm": null     // same from GPS, the only rate some aircraft send, use as fallback | live: geom_rate | history: point[11]
  },
  "nav": {
    "selected_altitude_ft": 12992      // altitude the crew dialed into the autopilot, null if not sent | live: nav_altitude_mcp | history: point[8].nav_altitude_mcp
  },
  "status": {
    "squawk": "3456",                  // ATC code, 7700 emergency, 7600 radio failure, 7500 hijack | live: squawk | history: point[8].squawk, reuse last
    "emergency": "none",               // none, general, lifeguard, minfuel, nordo, unlawful, downed | live: emergency | history: point[8].emergency, reuse last
    "seen_age_s": 0.487                // seconds since the last position, growing = losing contact | live: seen_pos | history: null
  },
  "anomaly": "none",                   // filled in by the AI model for the UI: none, engine_failure, aggressive_near_ground_speed, accelerating_descent, incoming_aircraft_collision, altitude_anomaly, squawk, heading_anomaly, heading_oscillation, route_deviation, signal_loss | live, history, sim: always "none", detector must ignore it on input
  "origin": "live"                     // live, history, or sim, debugging only, detector must ignore
}

{
  "icao24": "a1b2c3",                  // which aircraft, copied from its FlightState, the UI matches on this
  "flight_id": "UAL1842",              // callsign copied from its FlightState, for display
  "timestamp": 1727300123.25,          // time of the FlightState that triggered this, not when the detector ran
  "risk_score": 0.82,                  // overall danger from 0 to 1, combines all anomalies below
  "severity": "warning",               // risk as a label: normal < 0.30, advisory < 0.60, warning < 0.85, critical above
  "anomalies": [                       // one entry per problem found, empty if nothing is wrong
    {
      "type": "accelerating_descent",  // what kind of problem, any FlightState anomaly value except none
      "severity": 0.82,                // how confident this one check is, from 0 to 1
      "message": "Descent rate -4200 fpm (baro)"  // text for the alert card, includes the actual number that triggered it
    }
  ]
}