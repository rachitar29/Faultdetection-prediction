"""
Pydantic schemas for the predictive maintenance API.

AI4I sensor input and NASA C-MAPSS RUL input
are kept separate to prevent feature-schema mixing.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict


class EquipmentPredictionRequest(BaseModel):
    """Request for integrated equipment health prediction."""

    model_config = ConfigDict(extra="forbid")

    sensor_data: dict[str, Any] = Field(
        ...,
        description="AI4I-compatible sensor readings",
        min_length=1,
    )

    equipment_id: Optional[str] = Field(
        default=None,
        description="Optional equipment identifier.",
    )

    rul_sensor_data: Optional[dict[str, float]] = Field(
        default=None,
        description=(
            "Optional NASA C-MAPSS feature values. "
            "Must use the registered RUL model's feature schema."
        ),
    )


class EquipmentPredictionResponse(BaseModel):
    """Flexible response returned by the integrated service."""

    model_config = ConfigDict(extra="allow")

    success: bool = True
    equipment_id: Optional[str] = None

    failure_prediction: Optional[str] = None
    failure_probability: Optional[float] = None

    risk_level: Optional[str] = None
    risk_level_before_rul: Optional[str] = None
    rul_risk_level: Optional[str] = None
    rul_escalation_applied: bool = False

    classifier_risk_level: Optional[str] = None
    maintenance_priority: Optional[str] = None
    maintenance_recommendation: Optional[str] = None

    anomaly_detected: Optional[bool] = None
    anomaly_results: Optional[Any] = None
    anomaly_details: Optional[Any] = None
    anomaly_assessment: Optional[Any] = None

    rul_prediction: Optional[Any] = None
    rul_available: bool = False

    integrated_prediction: bool = True
    prediction_components: Optional[dict[str, bool]] = None

    recommendations: list[str] = Field(default_factory=list)
    explanations: dict[str, Any] = Field(default_factory=dict)
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None
    errors: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """API health-check response."""

    status: str
    service: str
    version: str