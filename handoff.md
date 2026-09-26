Syren
Syren is a real-time aviation anomaly detection and crisis-forecasting platform running locally on an NVIDIA Jetson Orin Nano. The system analyzes streams of aircraft telemetry, identifies abnormal behavior and developing hazards, and visualizes aircraft state and detected anomalies through a browser-based interface.
The project is being developed during a 36-hour hackathon by a three-person team. The priority is a functional end-to-end system with a strong visual demonstration rather than a large production architecture.
Core Concept
Syren represents each aircraft as a continuously changing flight state. Aircraft telemetry is passed through a detection engine that considers both the current state and recent history to identify potentially dangerous behavior.
The system should support three primary modes:
Simulation: Generate aircraft trajectories and allow anomalies or emergency conditions to be injected into otherwise normal flights.
Historical Replay: Replay telemetry from real historical flights through the same detection engine and visualize when Syren begins detecting abnormal behavior.
Fleet Backtesting: Process collections of historical normal and emergency flights to evaluate detection behavior, false alerts, detection lead time, and processing performance.
The same detection pipeline must operate regardless of where telemetry originates.
The fundamental architecture is:
Synthetic Simulation ─┐
│
Historical OpenSky ───┼──► Canonical Flight State ──► Detection Engine ──► API/WebSocket ──► UI
│
Historical FDR Data ──┘
Canonical Flight State
All data sources should be converted into a common representation before reaching the detection engine.
A flight state should contain fields such as:
{
"timestamp": 1727300123.25,
"flight_id": "UAL1842",

"position": {
"latitude": 28.43,
"longitude": -81.31,
"altitude_ft": 31000
},

"kinematics": {
"ground_speed_kts": 465,
"vertical_rate_fpm": -120,
"heading_deg": 247
},

"systems": {
"engine_status": null,
"hydraulic_status": null,
"electrical_status": null
},

"status": {
"squawk": "1200",
"emergency": false
}
}
The schema should allow optional fields because different sources provide different levels of telemetry. Missing information must remain missing rather than being synthesized.
Public ADS-B data generally provides position, altitude, velocity, heading, vertical rate, callsign, squawk, and related surveillance information. Richer aircraft parameters such as engine state, control inputs, pitch, roll, acceleration, or system warnings may only be available from Flight Data Recorder datasets or synthetic scenarios.
Historical Data
Real historical data is an important part of Syren.
The primary initial source is the OpenSky Network. OpenSky provides historical ADS-B/state-vector datasets and a reference dataset containing flights associated with in-flight emergency situations, including flights that transmitted the 7700 general-emergency transponder code.
Historical OpenSky flights should be converted into the canonical flight-state representation and replayed chronologically through Syren.
Emergency declaration information must be maintained separately as ground truth and must not be provided to the detector before it occurs.
This enables measurements such as:
First Syren warning: T+08:42
Syren critical alert: T+09:31
7700 observed: T+11:54

Warning lead time: 3m 12s
A 7700 transmission represents an observable emergency declaration, not necessarily the physical beginning of an emergency. Results should therefore be described as detection lead time relative to the declaration rather than claiming Syren knows exactly when the emergency began.
Historical normal flights should also be processed. Detecting emergencies is not meaningful if normal flights constantly generate alerts. Fleet backtesting should therefore eventually report metrics such as emergency detection rate, false alerts on normal flights, alert timing, and processing throughput.
Selected NTSB investigations may provide richer Flight Data Recorder telemetry as downloadable tabular data. Supporting these datasets is a secondary goal. Dataset-specific adapters should translate available FDR parameters into the same canonical flight representation without changing the detection engine.
Synthetic Simulation
Syren must not depend on historical data being available.
A synthetic simulator should generate normal aircraft behavior including cruise, climb, descent, approach, and other basic phases of flight.
The simulator should also support deliberate anomaly injection, including scenarios such as:
rapid or accelerating descent
abnormal speed loss
erratic speed changes
sudden heading deviations
altitude oscillation
unusual trajectory changes
emergency transponder states
simulated mechanical or system warnings
conflicting aircraft trajectories
The simulator creates the abnormal behavior but does not determine whether it is dangerous. That responsibility belongs exclusively to the detection engine.
Detection Engine
The detection engine receives chronological flight states and produces detection results.
Initial detection should prioritize understandable rule-based and temporal methods over complex machine-learning models.
Potential detectors include:
abnormal descent/climb
unusual speed behavior
heading instability
trajectory deviation
altitude instability
emergency squawk
system-warning combinations
aircraft-aircraft conflict prediction
Detection should consider recent history rather than treating each telemetry sample independently. For example, a rapidly accelerating descent is more significant than one isolated vertical-rate measurement.
Individual anomaly signals should contribute to an aggregate flight risk assessment.
A detection result may resemble:
{
"flight_id": "UAL1842",
"timestamp": 1727300123.25,
"risk_score": 0.82,
"severity": "critical",
"anomalies": [
{
"type": "ABNORMAL_DESCENT",
"severity": 0.91,
"message": "Accelerating abnormal descent detected"
},
{
"type": "SPEED_ANOMALY",
"severity": 0.54,
"message": "Uncharacteristic speed reduction detected"
}
]
}
Risk scores developed during the hackathon are prototype heuristics and should not be represented as calibrated real-world aviation safety probabilities.
Machine-learning anomaly detection can be added if time permits, particularly by learning normal flight behavior and identifying deviations, but ML is not required for the core system.
Jetson Orin Nano
The NVIDIA Jetson Orin Nano is the deployment and compute platform for Syren.
The Jetson runs headlessly. No monitor, keyboard, or mouse is required. Development occurs primarily on team members' laptops using Git, with one team member responsible for integrating and running the system on the Jetson.
The Jetson should run the simulation/replay and detection workloads locally. GPU/CUDA acceleration should be applied where it provides meaningful parallelism, particularly when evaluating large numbers of aircraft, trajectories, historical flights, or aircraft-aircraft interactions.
The project should be capable of demonstrating scaling across increasingly large numbers of aircraft or historical flight records.
The use of edge hardware is intended to demonstrate local processing without dependency on cloud compute during operation. Claims about latency, privacy, throughput, or reliability should be supported by what the prototype actually demonstrates.
Backend and Communication
The backend should remain simple.
A small API layer, such as FastAPI, can expose simulation controls and stream state to the browser.
Control operations can use ordinary HTTP endpoints:
POST /api/start
POST /api/pause
POST /api/reset
POST /api/scenario
POST /api/speed
Continuous aircraft and detection state should be streamed through a WebSocket:
/ws/state
The internal simulation may operate at a much higher frequency than the UI. The backend should publish snapshots at approximately 10 to 20 Hz rather than sending every simulation timestep.
The browser can interpolate between snapshots to provide smooth visual motion.
User Interface
The final interface is browser-based and is viewed from a laptop connected to the Jetson over the network. The Jetson itself does not require a display.
The primary visualization should resemble an aviation monitoring/control interface centered around a map containing active aircraft.
Selecting an aircraft should expose:
current altitude
speed
heading
vertical rate
current risk state
detected anomalies
recent telemetry history
risk history
The interface should prominently surface warning and critical aircraft.
Historical replay should include timeline controls, playback speed, telemetry graphs, risk evolution, anomaly markers, and known ground-truth events such as an emergency declaration.
The UI should initially operate entirely against mocked flight and detection data so frontend development can proceed independently from the simulator and detector.
Development Structure
The repository should maintain clear separation between major components:
syren/
├── frontend/
│ └── browser visualization
│
├── backend/
│ └── API and WebSocket communication
│
├── simulation/
│ ├── synthetic flight generation
│ ├── scenario injection
│ └── historical replay
│
├── detection/
│ ├── individual anomaly detectors
│ ├── temporal analysis
│ └── risk aggregation
│
├── data/
│ ├── dataset adapters
│ └── historical datasets
│
├── shared/
│ └── common models/schema
│
└── docs/
└── protocol and architecture documentation
The canonical flight-state and detection-result contracts should be established first. Simulation, detection, and frontend development should then proceed asynchronously against those contracts.
The critical end-to-end path is:
Flight telemetry
↓
Canonical FlightState
↓
Detection Engine
↓
DetectionResult
↓
Backend
↓
WebSocket
↓
Browser visualization
Historical datasets, synthetic simulations, and future live telemetry should all enter through the same interface.
Hackathon Priorities
The minimum successful Syren demonstration is an end-to-end system in which aircraft telemetry is replayed or simulated, analyzed on the Jetson, and visualized in real time in the browser with developing anomalies clearly identified.
Real historical emergency replay is a high priority because it provides a stronger demonstration than purely synthetic failures.
The desired final demonstration includes:
normal aircraft moving through the visualization
synthetic anomaly injection
real historical flight replay
temporal anomaly detection
evolving flight risk
clear warning/critical alerts
comparison of Syren detection timing against known historical events
processing of normal historical flights to demonstrate false-alert behavior
Jetson-local execution
performance/scaling measurements if time permits
The project should favor a reliable, understandable end-to-end implementation over unnecessary infrastructure or excessive model complexity.
