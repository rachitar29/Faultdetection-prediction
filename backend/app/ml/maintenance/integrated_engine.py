
"""
Unified Equipment Prediction Engine.

Combines:
- Supervised failure prediction
- Anomaly detection
- Risk assessment
- SHAP explanations
- Optional NASA C-MAPSS RUL prediction

AI4I and NASA C-MAPSS inputs remain separate.
RUL risk escalation uses configurable project heuristics,
not certified industrial safety limits.
"""

from app.ml.maintenance.engine import predict_equipment_health

from app.ml.maintenance.anomaly_service import (
    detect_machine_anomalies,
    get_anomaly_flags,
)

from app.ml.maintenance.risk_assessment import (
    assess_equipment_risk,
)

from app.ml.maintenance.rul_prediction import (
    predict_single_machine_rul,
)


# ==========================================
# CONFIGURATION
# ==========================================

RISK_LEVELS = [
    "Low",
    "Medium",
    "High",
    "Critical"
]


RUL_CATEGORY_TO_RISK = {
    "LOW": "Low",
    "MEDIUM": "Medium",
    "HIGH": "High",
    "CRITICAL": "Critical",
}


MAINTENANCE_ACTIONS = {

    "Low": (
        "Routine",
        "Continue routine monitoring and follow the "
        "manufacturer's maintenance schedule."
    ),

    "Medium": (
        "Moderate",
        "Increase monitoring frequency and review "
        "recent sensor readings."
    ),

    "High": (
        "High",
        "Arrange an equipment inspection and investigate "
        "the detected abnormal operating conditions."
    ),

    "Critical": (
        "Urgent",
        "Arrange urgent equipment inspection. Follow "
        "approved site safety procedures."
    ),
}


# ==========================================
# RUL RISK ESCALATION
# ==========================================

def apply_rul_risk_escalation(
    current_risk,
    rul_prediction
):
    """
    Escalate risk when the RUL category indicates
    greater urgency.

    Risk can increase because of RUL but can never decrease.
    """

    if current_risk not in RISK_LEVELS:
        raise ValueError(
            f"Unknown current risk level: {current_risk}"
        )

    if not isinstance(rul_prediction, dict):
        raise TypeError(
            "RUL prediction must be a dictionary."
        )

    category = rul_prediction.get("rul_category")

    if not isinstance(category, str):
        raise ValueError(
            "RUL prediction has no valid rul_category."
        )

    category = category.strip().upper()

    if category not in RUL_CATEGORY_TO_RISK:
        raise ValueError(
            f"Unknown RUL category: {category}"
        )

    rul_risk = RUL_CATEGORY_TO_RISK[category]

    current_index = RISK_LEVELS.index(
        current_risk
    )

    rul_index = RISK_LEVELS.index(
        rul_risk
    )

    final_risk = RISK_LEVELS[
        max(current_index, rul_index)
    ]

    return {
        "risk_level_before_rul": current_risk,

        "rul_risk_level": rul_risk,

        "risk_level": final_risk,

        "rul_escalation_applied": (
            final_risk != current_risk
        ),
    }


# ==========================================
# INTEGRATED EQUIPMENT PREDICTION
# ==========================================

def predict_equipment_condition(
    sensor_data,
    rul_sensor_data=None
):
    """
    Generate integrated equipment predictions.

    sensor_data:
        AI4I-compatible sensor readings.

    rul_sensor_data:
        Separate NASA C-MAPSS-compatible input.

    Never pass AI4I sensor data to the NASA RUL model.
    """

    # ======================================
    # STEP 1: SUPERVISED FAILURE PREDICTION
    # ======================================

    prediction = predict_equipment_health(
        sensor_data
    )

    if not isinstance(prediction, dict):
        raise ValueError(
            "Failure prediction returned an invalid result."
        )

    if "failure_probability" not in prediction:
        raise ValueError(
            "Failure prediction is missing failure_probability."
        )

    # ======================================
    # STEP 2: ANOMALY DETECTION
    # ======================================

    anomaly_results = detect_machine_anomalies(
        sensor_data
    )

    # ======================================
    # STEP 3: EXTRACT ANOMALY FLAGS
    # ======================================

    anomaly_flags = get_anomaly_flags(
        anomaly_results
    )

    # ======================================
    # STEP 4: COMBINED RISK ASSESSMENT
    # ======================================

    risk_assessment = assess_equipment_risk(
        failure_probability=prediction["failure_probability"],
        anomaly_results=anomaly_flags,
    )

    result = prediction.copy()

    result["classifier_risk_level"] = (
        risk_assessment["classifier_risk_level"]
    )

    result["risk_level"] = (
        risk_assessment["overall_risk_level"]
    )

    result["maintenance_priority"] = (
        risk_assessment["maintenance_priority"]
    )

    result["maintenance_recommendation"] = (
        risk_assessment["maintenance_recommendation"]
    )

    result["anomaly_results"] = anomaly_results

    result["anomaly_assessment"] = (
        risk_assessment["anomaly_assessment"]
    )

    result["risk_assessment_method"] = (
        risk_assessment["assessment_method"]
    )

    # ======================================
    # STEP 5: OPTIONAL NASA RUL PREDICTION
    # ======================================

    if rul_sensor_data is not None:

        rul_result = predict_single_machine_rul(
            rul_sensor_data
        )

        if not isinstance(rul_result, dict):
            raise ValueError(
                "RUL prediction returned an invalid result."
            )

        escalation = apply_rul_risk_escalation(
            current_risk=result["risk_level"],
            rul_prediction=rul_result,
        )

        result["rul_prediction"] = rul_result

        result["rul_available"] = True

        result.update(escalation)

        # Update maintenance recommendation only
        # when RUL increases the overall risk.

        if escalation["rul_escalation_applied"]:

            priority, recommendation = MAINTENANCE_ACTIONS[
                escalation["risk_level"]
            ]

            result["maintenance_priority"] = priority

            result["maintenance_recommendation"] = (
                recommendation
            )

        result["risk_assessment_method"] += (
            " RUL-based escalation heuristic also applied."
        )

    else:

        result["rul_prediction"] = None

        result["rul_available"] = False

        result["risk_level_before_rul"] = (
            result["risk_level"]
        )

        result["rul_risk_level"] = None

        result["rul_escalation_applied"] = False

    # ======================================
    # STEP 6: EXPLANATION STATUS
    # ======================================

    explanation_status = prediction.get(
        "explanation_status",
        "unavailable"
    )

    shap_available = (
        explanation_status == "available"
    )

    result["shap_explanation_status"] = (
        explanation_status
    )

    # ======================================
    # STEP 7: COMPONENT METADATA
    # ======================================

    result["integrated_prediction"] = True

    result["prediction_components"] = {

        "failure_classification": True,

        "anomaly_detection": True,

        "combined_risk_assessment": True,

        "shap_explanation": shap_available,

        "remaining_useful_life": (
            rul_sensor_data is not None
        ),

        "rul_risk_escalation": (
            rul_sensor_data is not None
        ),
    }

    return result
