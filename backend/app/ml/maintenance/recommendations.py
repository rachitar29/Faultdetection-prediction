
from app.ml.maintenance.risk_engine import calculate_machine_risk


RECOMMENDATIONS = {
    "CRITICAL": {
        "priority": "IMMEDIATE",
        "actions": [
            "Inspect the machine immediately.",
            "Consider controlled shutdown according to safety procedures.",
            "Check vibration, temperature, pressure and electrical readings.",
            "Arrange maintenance personnel for urgent inspection."
        ]
    },
    "HIGH": {
        "priority": "URGENT",
        "actions": [
            "Schedule preventive maintenance soon.",
            "Inspect bearings, motors and moving components.",
            "Review recent sensor trends.",
            "Increase monitoring frequency."
        ]
    },
    "MEDIUM": {
        "priority": "MODERATE",
        "actions": [
            "Monitor sensor readings regularly.",
            "Inspect the machine during the next maintenance window.",
            "Check for abnormal vibration or temperature changes."
        ]
    },
    "LOW": {
        "priority": "ROUTINE",
        "actions": [
            "Continue normal monitoring.",
            "Follow the regular maintenance schedule.",
            "Record sensor readings for future trend analysis."
        ]
    }
}


def generate_maintenance_recommendations(model_name, sensor_data):
    """Generate recommendations from predicted failure risk."""

    risk_result = calculate_machine_risk(
        model_name,
        sensor_data
    )

    recommendations = []

    for prediction in risk_result["predictions"]:
        category = prediction["risk_category"]
        recommendation = RECOMMENDATIONS[category]

        recommendations.append({
            **prediction,
            "maintenance_priority": recommendation["priority"],
            "recommended_actions": recommendation["actions"]
        })

    return {
        "model_name": model_name,
        "recommendations": recommendations
    }
