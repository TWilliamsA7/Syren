{
  "timestamp": 1727300123.25,          // when the position was measured, Unix seconds UTC | live: resp["now"]/1000 - ac["seen_pos"] | history: file["timestamp"] + point[0]
  "icao24": "a1b2c3",                  // aircraft's permanent transponder ID, use this to match records | live: ac["hex"] | history: file["icao"]
  "flight_id": "UAL1842",              // flight callsign, falls back to icao24 if missing | live: ac["flight"] | history: point[8]["flight"], reuse last one if null
  "aircraft": {
    "registration": "N37522",          // tail number painted on the plane | live: ac["r"] | history: file["r"]
    "type_code": "B38M",               // aircraft model code, B38M = Boeing 737 MAX 8 | live: ac["t"] | history: file["t"]
    "category": "A3"                   // size class, A1 light ... A3 large ... A5 heavy | live: ac["category"] | history: point[8]["category"], reuse last one if null
  },
  "position": {
    "latitude": 28.43,                 // GPS latitude in degrees, positive = north | live: ac["lat"] | history: point[1]
    "longitude": -81.31,               // GPS longitude in degrees, negative = west | live: ac["lon"] | history: point[2]
    "altitude_baro_ft": 31000,         // altitude from air pressure in feet, what ATC uses | live: ac["alt_baro"] | history: point[3]
    "altitude_geom_ft": 31240,         // altitude from GPS in feet, closer to true height | live: ac["alt_geom"] | history: point[10]
    "on_ground": false,                // true when the plane is on the ground | live: ac["alt_baro"] == "ground" | history: point[3] == "ground"
    "source": "adsb_icao",             // adsb_icao = plane's own GPS, mlat = estimated by ground receivers (less accurate) | live: ac["type"] | history: point[9]
    "stale": false                     // true if no position was heard for 20+ seconds before this one | live: ac["seen_pos"] > 20 | history: point[6] & 1
  },
  "kinematics": {
    "ground_speed_kts": 465.0,         // speed over the ground in knots, affected by wind | live: ac["gs"] | history: point[4]
    "track_deg": 247.0,                // direction of travel in degrees, 0 = north, 90 = east | live: ac["track"] | history: point[5]
    "vertical_rate_baro_fpm": -128,    // climb (+) or descent (-) in feet per minute, from air pressure | live: ac["baro_rate"] | history: point[7]
    "vertical_rate_geom_fpm": -128,    // climb (+) or descent (-) in feet per minute, from GPS | live: ac["geom_rate"] | history: point[11]
    "ias_kts": 285,                    // airspeed through the air in knots, not affected by wind, often missing in the US | live: ac["ias"] | history: point[12]
    "roll_deg": -0.4                   // bank angle in degrees, negative = banking left, often missing in the US | live: ac["roll"] | history: point[13]
  },
  "status": {
    "squawk": "4521",                  // 4-digit code set by the pilot, 7700 = emergency, 7600 = radio failure, 7500 = hijack | live: ac["squawk"] | history: point[8]["squawk"], reuse last one if null
    "emergency": "none",               // declared emergency: none, general, lifeguard, minfuel, nordo, unlawful, downed | live: ac["emergency"] | history: point[8]["emergency"], reuse last one if null
    "seen_age_s": 0.8                  // seconds since the last position was heard, growing = losing contact | live: ac["seen_pos"] | history: always null
  },
  "origin": "live"                     // where this record came from: live, history, or sim, for debugging only
}

{
  "icao24": "a1b2c3",                  // which aircraft, copied from its FlightState, the UI matches on this
  "flight_id": "UAL1842",              // callsign copied from its FlightState, for display
  "timestamp": 1727300123.25,          // time of the FlightState that triggered this, not when the detector ran
  "risk_score": 0.82,                  // overall danger from 0 to 1, combines all anomalies below
  "severity": "warning",               // risk as a label: normal < 0.30, advisory < 0.60, warning < 0.85, critical above
  "anomalies": [                       // one entry per problem found, empty if nothing is wrong
    {
      "type": "RAPID_DESCENT",         // what kind of problem, from a fixed list the UI knows
      "severity": 0.82,                // how confident this one check is, from 0 to 1
      "message": "Descent rate -4200 fpm (baro)"  // text for the alert card, includes the actual number that triggered it
    }
  ]
}