"""Rules for emergency transponder squawk codes."""

from detection.models import Anomaly, FlightState
from detection.detectors.base import RuleDetector


class EmergencySquawkDetector(RuleDetector):
    _CODES = {
        "7500": (0.99, "Unlawful-interference squawk 7500"),
        "7600": (0.90, "Radio-failure squawk 7600"),
        "7700": (0.98, "General-emergency squawk 7700"),
    }

    def update(self, state: FlightState) -> Anomaly | None:
        code = (state.status.squawk or "").strip()
        match = self._CODES.get(code)
        if match is None:
            return None
        severity, message = match
        return Anomaly("EMERGENCY_SQUAWK", severity, message)
