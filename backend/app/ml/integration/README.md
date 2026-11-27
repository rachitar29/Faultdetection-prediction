
# Equipment Fault Detection - ML Integration

## 1. Overview

This package provides a centralized interface to the equipment
fault detection and predictive maintenance ML system.

It connects existing trained models and prediction modules
without duplicating their implementation.

## 2. Main Entry Point

File: `app/ml/integration/ml_service.py`

Import:

```python
from app.ml.integration.ml_service import MLService

ml_service = MLService()
```

## 3. Prediction Input

The service accepts a dictionary or MLPredictionRequest.

Example:

```python
request = {
    "equipment_id": "MACHINE-001",
    "sensor_data": {
        "Air temperature [K]": 300.5,
        "Process temperature [K]": 310.2,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40,
        "Tool wear [min]": 120,
        "Type": "M"
    }
}

result = ml_service.predict(request)
```

## 4. Prediction Components

The unified engine integrates:

1. XGBoost supervised failure prediction.
2. Isolation Forest anomaly detection.
3. Local Outlier Factor anomaly detection.
4. Combined risk assessment.
5. SHAP feature explanations.
6. Maintenance recommendations.
7. Optional NASA C-MAPSS RUL prediction.

## 5. Output Structure

The service returns:

- success
- equipment_id
- failure_prediction
- failure_probability
- risk_level
- anomaly_detected
- anomaly_details
- rul_prediction
- rul_available
- recommendations
- explanations
- model_metadata
- timestamp
- errors

## 6. Risk Assessment

Risk categories:

- Low
- Medium
- High
- Critical

Risk assessment uses classifier failure probability and
anomaly consensus.

When both anomaly detectors agree, the risk category is
escalated by one level.

This is a configurable project heuristic, not an
industrial safety standard.

## 7. Anomaly Detection

Both anomaly detectors return:

- is_anomaly
- anomaly_score
- optimized_threshold
- threshold_source

Anomaly scores are not probabilities.

## 8. RUL Prediction

NASA C-MAPSS RUL prediction is optional.

It requires a separate NASA-compatible sensor input.

Never pass AI4I sensor readings to the NASA RUL model.

## 9. Model and Dataset Compatibility

The current primary failure classifier is trained for
AI4I-compatible sensor inputs.

The ML service does not support arbitrary sensor schemas
without a compatible preprocessing and model pipeline.

## 10. Testing

Run mocked integration tests:

```powershell
python -m unittest app.ml.integration.test_ml_service -v
```

Run actual model smoke test:

```powershell
python -m app.ml.integration.test_real_prediction
```

## 11. Integration Boundary

This package belongs to the ML/Data Engineering module.

It does not implement:

- FastAPI endpoints
- Database persistence
- Frontend components
- Authentication

These remain outside the ML integration package.
