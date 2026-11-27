
"""
Central ML service interface.

Connects the integration layer to the existing unified
equipment prediction engine.

No model training or duplicate prediction logic is implemented here.
"""

from typing import Any, Dict, Union

from app.ml.integration.schemas import MLPredictionRequest
from app.ml.integration.response_formatter import (
    format_prediction_response,
)
from app.ml.maintenance.integrated_engine import (
    predict_equipment_condition,
)


class MLService:
    """Main interface for equipment ML predictions."""

    def predict(
        self,
        request: Union[MLPredictionRequest, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate an integrated equipment prediction.

        Required:
            sensor_data: AI4I-compatible sensor readings.

        Optional:
            equipment_id: Equipment identifier.
            rul_sensor_data: Separate NASA-compatible input.

        Returns:
            JSON-compatible dictionary containing ML results.
        """

        # STEP 1: Validate request type.
        if isinstance(request, dict):
            request = MLPredictionRequest(**request)

        if not isinstance(request, MLPredictionRequest):
            raise TypeError(
                "Request must be an MLPredictionRequest "
                "or dictionary."
            )

        # STEP 2: Validate sensor input.
        if not isinstance(request.sensor_data, dict):
            raise TypeError(
                "sensor_data must be a dictionary."
            )

        if not request.sensor_data:
            raise ValueError(
                "sensor_data cannot be empty."
            )

        if request.rul_sensor_data is not None:
            if not isinstance(request.rul_sensor_data, dict):
                raise TypeError(
                    "rul_sensor_data must be a dictionary or None."
                )

            if not request.rul_sensor_data:
                raise ValueError(
                    "rul_sensor_data cannot be an empty dictionary."
                )

        # STEP 3: Call the existing integrated ML engine.
        raw_prediction = predict_equipment_condition(
            sensor_data=request.sensor_data,
            rul_sensor_data=request.rul_sensor_data,
        )

        # STEP 4: Validate engine output.
        if not isinstance(raw_prediction, dict):
            raise TypeError(
                "The integrated prediction engine "
                "must return a dictionary."
            )

        # STEP 5: Format the response.
        formatted_response = format_prediction_response(
            prediction=raw_prediction,
            equipment_id=request.equipment_id,
        )

        return formatted_response


# Reusable service instance.
ml_service = MLService()
