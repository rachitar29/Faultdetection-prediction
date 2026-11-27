"""
Shared data contracts for the ML integration layer.

Independent of FastAPI and database implementation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MLPredictionRequest:
    """Input received by the ML prediction service."""

    sensor_data: Dict[str, Any]

    equipment_id: Optional[str] = None
    machine_type: Optional[str] = None

    include_anomaly: bool = True
    include_explanation: bool = True

    # Separate NASA C-MAPSS compatible input.
    # Never pass AI4I sensor data to the NASA RUL model.
    rul_sensor_data: Optional[Dict[str, Any]] = None


@dataclass
class MLPredictionResponse:
    """Standardized output returned by the ML service."""

    success: bool

    equipment_id: Optional[str] = None

    failure_prediction: Optional[str] = None
    failure_probability: Optional[float] = None

    risk_level: Optional[str] = None

    anomaly_detected: Optional[bool] = None
    anomaly_details: Dict[str, Any] = field(default_factory=dict)

    rul_prediction: Any = None
    rul_available: bool = False

    # Explicit RUL escalation information.
    risk_level_before_rul: Optional[str] = None
    rul_risk_level: Optional[str] = None
    rul_escalation_applied: bool = False

    recommendations: List[str] = field(default_factory=list)
    explanations: Dict[str, Any] = field(default_factory=dict)

    model_metadata: Dict[str, Any] = field(default_factory=dict)

    timestamp: Optional[str] = None
    errors: List[str] = field(default_factory=list)