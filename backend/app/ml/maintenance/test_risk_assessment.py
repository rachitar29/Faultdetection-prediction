import pytest

from app.ml.maintenance.risk_assessment import (
    assess_equipment_risk,
    classify_failure_probability,
)


ALL_CLEAR = {
    "Isolation Forest": False,
    "Local Outlier Factor": False,
    "One-Class SVM": False,
}


def make_anomalies(*flagged):
    return {
        "Isolation Forest": "Isolation Forest" in flagged,
        "Local Outlier Factor": "Local Outlier Factor" in flagged,
        "One-Class SVM": "One-Class SVM" in flagged,
    }


@pytest.mark.parametrize(
    "probability, expected",
    [
        (0.10, "Low"),
        (0.25, "Medium"),
        (0.30, "Medium"),
        (0.50, "High"),
        (0.75, "Critical"),
        (0.95, "Critical"),
    ],
)
def test_failure_probability_classification(probability, expected):
    assert classify_failure_probability(probability) == expected


def test_normal_machine_has_low_risk():
    result = assess_equipment_risk(
        failure_probability=0.10,
        anomaly_results=ALL_CLEAR,
    )

    assert result["classifier_risk_level"] == "Low"
    assert result["overall_risk_level"] == "Low"
    assert result["anomaly_assessment"]["detectors_flagged"] == 0


def test_one_anomaly_does_not_escalate_risk():
    result = assess_equipment_risk(
        failure_probability=0.30,
        anomaly_results=make_anomalies("Isolation Forest"),
    )

    assert result["classifier_risk_level"] == "Medium"
    assert result["overall_risk_level"] == "Medium"
    assert result["anomaly_assessment"]["detectors_flagged"] == 1


def test_majority_anomaly_consensus_escalates_risk_once():
    result = assess_equipment_risk(
        failure_probability=0.30,
        anomaly_results=make_anomalies(
            "Isolation Forest",
            "Local Outlier Factor",
        ),
    )

    assert result["classifier_risk_level"] == "Medium"
    assert result["overall_risk_level"] == "High"
    assert result["anomaly_assessment"]["detectors_flagged"] == 2


def test_all_three_anomalies_escalate_risk_once():
    result = assess_equipment_risk(
        failure_probability=0.10,
        anomaly_results=make_anomalies(
            "Isolation Forest",
            "Local Outlier Factor",
            "One-Class SVM",
        ),
    )

    assert result["classifier_risk_level"] == "Low"
    assert result["overall_risk_level"] == "Medium"
    assert result["anomaly_assessment"]["detectors_flagged"] == 3


def test_critical_risk_cannot_escalate_beyond_critical():
    result = assess_equipment_risk(
        failure_probability=0.90,
        anomaly_results=make_anomalies(
            "Isolation Forest",
            "Local Outlier Factor",
        ),
    )

    assert result["classifier_risk_level"] == "Critical"
    assert result["overall_risk_level"] == "Critical"


def test_missing_detector_is_rejected():
    incomplete = {
        "Isolation Forest": False,
        "Local Outlier Factor": False,
    }

    with pytest.raises(ValueError, match="Missing detector results"):
        assess_equipment_risk(
            failure_probability=0.10,
            anomaly_results=incomplete,
        )


@pytest.mark.parametrize(
    "probability",
    [-0.01, 1.01, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_failure_probability_is_rejected(probability):
    with pytest.raises(ValueError):
        assess_equipment_risk(
            failure_probability=probability,
            anomaly_results=ALL_CLEAR,
        )
