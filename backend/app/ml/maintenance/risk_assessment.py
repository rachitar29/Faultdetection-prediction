
"""
Equipment Risk Assessment Module

Combines:
1. Supervised failure probability.
2. Isolation Forest anomaly detection.
3. Local Outlier Factor anomaly detection.
4. One-Class SVM anomaly detection.

Anomaly scores are not probabilities. The service supplies
boolean anomaly flags using validation thresholds.

All three detector results are required for a complete
anomaly consensus assessment.

Risk escalation is a configurable project heuristic,
not an industrial safety standard.
"""

from datetime import datetime, timezone
import math

import numpy as np


# ==========================================
# CONFIGURATION
# ==========================================

RISK_LEVELS = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

FAILURE_PROBABILITY_THRESHOLDS = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75
}

ANOMALY_DETECTORS = [
    "Isolation Forest",
    "Local Outlier Factor",
    "One-Class SVM"
]


# ==========================================
# CLASSIFY FAILURE PROBABILITY
# ==========================================

def classify_failure_probability(probability):
    """Classify equipment risk using failure probability."""

    try:
        probability = float(probability)
    except (TypeError, ValueError):
        raise ValueError(
            "Failure probability must be numeric."
        )

    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(
            "Failure probability must be between 0 and 1."
        )

    if probability < FAILURE_PROBABILITY_THRESHOLDS["low"]:
        return "Low"

    if probability < FAILURE_PROBABILITY_THRESHOLDS["medium"]:
        return "Medium"

    if probability < FAILURE_PROBABILITY_THRESHOLDS["high"]:
        return "High"

    return "Critical"


# ==========================================
# VALIDATE ANOMALY RESULTS
# ==========================================

def validate_anomaly_results(anomaly_results):
    """
    Validate that all three anomaly detectors returned
    explicit boolean results.

    Supports:
    - Python bool
    - NumPy bool_

    Missing results and invalid values are rejected.
    """

    if not isinstance(anomaly_results, dict):
        raise TypeError(
            "Anomaly results must be a dictionary containing "
            "all three detector results."
        )

    missing_detectors = [
        model_name
        for model_name in ANOMALY_DETECTORS
        if model_name not in anomaly_results
    ]

    if missing_detectors:
        raise ValueError(
            "Incomplete anomaly assessment. Missing detector "
            "results: " + ", ".join(missing_detectors)
        )

    validated = {}

    for model_name in ANOMALY_DETECTORS:

        result = anomaly_results[model_name]

        # Support both service response dictionaries
        # and already-extracted boolean flags.

        if isinstance(result, dict):

            if "is_anomaly" not in result:
                raise ValueError(
                    f"{model_name} result is missing the "
                    "'is_anomaly' field."
                )

            result = result["is_anomaly"]

        # Accept Python bool and NumPy bool_.
        if not isinstance(result, (bool, np.bool_)):
            raise ValueError(
                f"{model_name} anomaly result must be boolean."
            )

        # Convert NumPy boolean into standard Python boolean.
        validated[model_name] = bool(result)

    return validated


# ==========================================
# ANOMALY CONSENSUS
# ==========================================

def determine_anomaly_status(anomaly_results):
    """Determine agreement between the three anomaly detectors."""

    anomaly_results = validate_anomaly_results(
        anomaly_results
    )

    detected_count = sum(
        anomaly_results.values()
    )

    total_detectors = len(ANOMALY_DETECTORS)

    if detected_count == 0:

        consensus_level = "None"

        status = (
            "No anomaly detector identified an anomaly."
        )

    elif detected_count == 1:

        consensus_level = "Single"

        status = (
            "One anomaly detector identified unusual "
            "sensor behaviour."
        )

    elif detected_count == 2:

        consensus_level = "Majority"

        status = (
            "Two of three anomaly detectors identified "
            "unusual sensor behaviour."
        )

    else:

        consensus_level = "Unanimous"

        status = (
            "All three anomaly detectors identified "
            "unusual sensor behaviour."
        )

    return {
        "isolation_forest_anomaly": (
            anomaly_results["Isolation Forest"]
        ),

        "lof_anomaly": (
            anomaly_results["Local Outlier Factor"]
        ),

        "one_class_svm_anomaly": (
            anomaly_results["One-Class SVM"]
        ),

        "detectors_flagged": detected_count,

        "total_detectors": total_detectors,

        "anomaly_consensus_level": consensus_level,

        "anomaly_consensus": status
    }


# ==========================================
# PROBABILITY INPUT VALIDATION
# ==========================================

def classify_probability_input(probability):
    """Convert and validate probability input."""

    try:
        probability = float(probability)
    except (TypeError, ValueError):
        raise ValueError(
            "Failure probability must be numeric."
        )

    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(
            "Failure probability must be between 0 and 1."
        )

    return probability


# ==========================================
# OVERALL RISK ASSESSMENT
# ==========================================

def assess_equipment_risk(
    failure_probability,
    anomaly_results
):
    """
    Assess equipment risk using supervised failure
    probability and three anomaly detectors.

    A majority agreement (at least two detectors)
    escalates classifier risk by one category.

    Incomplete anomaly results are rejected.

    The original classifier risk is always preserved.
    """

    # STEP 1: Validate failure probability.

    failure_probability = classify_probability_input(
        failure_probability
    )

    # STEP 2: Classify supervised model risk.

    classifier_risk = classify_failure_probability(
        failure_probability
    )

    # STEP 3: Evaluate anomaly agreement.

    anomaly_info = determine_anomaly_status(
        anomaly_results
    )

    # STEP 4: Determine overall risk.

    overall_risk = classifier_risk

    if anomaly_info["detectors_flagged"] >= 2:

        current_index = RISK_LEVELS.index(
            classifier_risk
        )

        overall_risk = RISK_LEVELS[
            min(
                current_index + 1,
                len(RISK_LEVELS) - 1
            )
        ]

    # STEP 5: Determine maintenance priority.

    priority_map = {
        "Low": "Routine",
        "Medium": "Moderate",
        "High": "High",
        "Critical": "Urgent"
    }

    action_map = {
        "Low": (
            "Continue routine monitoring and follow "
            "the manufacturer's maintenance schedule."
        ),

        "Medium": (
            "Increase monitoring frequency and review "
            "recent sensor readings."
        ),

        "High": (
            "Arrange an equipment inspection and investigate "
            "the detected abnormal operating conditions."
        ),

        "Critical": (
            "Arrange urgent equipment inspection. Follow "
            "approved site safety procedures."
        )
    }

    # STEP 6: Return complete risk assessment.

    return {
        "classifier_risk_level": classifier_risk,

        "overall_risk_level": overall_risk,

        "failure_probability": round(
            failure_probability,
            4
        ),

        "failure_probability_percent": round(
            failure_probability * 100,
            2
        ),

        "anomaly_assessment": anomaly_info,

        "maintenance_priority": (
            priority_map[overall_risk]
        ),

        "maintenance_recommendation": (
            action_map[overall_risk]
        ),

        "assessment_method": (
            "Supervised failure probability with "
            "three-detector majority-consensus "
            "anomaly escalation heuristic."
        ),

        "assessment_timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }
