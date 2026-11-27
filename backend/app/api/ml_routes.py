
"""
FastAPI routes for AI predictive maintenance.

Includes:
- Single-machine prediction through MLService
- Row-wise CSV batch prediction through the integrated prediction engine
- Separate downloads for successful predictions and rejected rows

Existing registered models are loaded only.
"""

import logging
from io import BytesIO, StringIO

import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    File,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse

from app.ml.integration.ml_service import ml_service
from app.ml.ingestion.batch_prediction_service import (
    predict_batch_dataframe,
)
from app.schemas.ml_schemas import (
    EquipmentPredictionRequest,
    HealthResponse,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/ml",
    tags=["Predictive Maintenance"],
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 64 * 1024
PREVIEW_ROWS = 100


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "service": "AI Predictive Maintenance API",
        "version": "1.0.0",
    }


@router.post("/predict")
def predict_equipment(request: EquipmentPredictionRequest):
    """Predict the condition of one equipment reading."""

    try:
        result = ml_service.predict({
            "sensor_data": request.sensor_data,
            "equipment_id": request.equipment_id,
            "rul_sensor_data": request.rul_sensor_data,
        })

        return JSONResponse(
            content=jsonable_encoder(result)
        )

    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        logger.exception(
            "Required prediction model or artifact was not found."
        )

        raise HTTPException(
            status_code=503,
            detail="A required prediction model is unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected single-equipment prediction failure."
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal server error.",
        ) from exc


async def _process_csv_upload(file: UploadFile):
    """
    Read a CSV upload with a bounded memory limit.

    The limit is enforced while reading, not after the complete
    upload has already been loaded into memory.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please select a CSV file.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only .csv files are supported.",
        )

    content = bytearray()

    while True:
        chunk = await file.read(UPLOAD_CHUNK_BYTES)

        if not chunk:
            break

        content.extend(chunk)

        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 25 MB upload limit.",
            )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        dataframe = pd.read_csv(
            BytesIO(content),
            encoding="utf-8-sig",
        )

    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not read the uploaded CSV file.",
        ) from exc

    except pd.errors.EmptyDataError as exc:
        raise HTTPException(
            status_code=422,
            detail="The uploaded CSV contains no readable data.",
        ) from exc

    if dataframe.empty:
        raise HTTPException(
            status_code=422,
            detail="The uploaded CSV contains no records.",
        )

    if dataframe.columns.duplicated().any():
        raise HTTPException(
            status_code=422,
            detail="The uploaded CSV contains duplicate column names.",
        )

    return dataframe


def _dataframe_records(dataframe, limit=None):
    """Convert a DataFrame into JSON-safe records."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Batch service returned an invalid DataFrame."
        )

    selected = (
        dataframe.head(limit)
        if limit is not None
        else dataframe
    )

    selected = selected.astype(object).where(
        pd.notna(selected),
        None,
    )

    return jsonable_encoder(
        selected.to_dict(orient="records")
    )


def _run_batch_prediction(dataframe):
    """Run the canonical row-wise batch prediction service."""

    result = predict_batch_dataframe(dataframe)

    if not isinstance(result, dict):
        raise TypeError(
            "Batch prediction service returned an invalid result."
        )

    predictions = result.get("predictions")
    rejected_rows = result.get("rejected_rows")
    summary = result.get("summary")

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError(
            "Batch prediction result is missing its predictions DataFrame."
        )

    if not isinstance(rejected_rows, pd.DataFrame):
        raise TypeError(
            "Batch prediction result is missing its rejected_rows DataFrame."
        )

    if not isinstance(summary, dict):
        raise TypeError(
            "Batch prediction result is missing its summary."
        )

    return predictions, rejected_rows, summary


def _handle_batch_exception(exc):
    """Convert batch-service exceptions into safe HTTP errors."""

    if isinstance(exc, (ValueError, TypeError, KeyError)):
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    if isinstance(exc, FileNotFoundError):
        logger.exception(
            "Batch prediction model or artifact was not found."
        )

        raise HTTPException(
            status_code=503,
            detail="A required prediction model is unavailable.",
        ) from exc

    logger.exception(
        "Unexpected batch prediction failure."
    )

    raise HTTPException(
        status_code=500,
        detail="Batch prediction failed due to an internal server error.",
    ) from exc


@router.post("/predict-csv")
async def predict_csv(
    file: UploadFile = File(...),
):
    """
    Upload a sensor CSV and return prediction and rejection previews.

    Both successful and rejected rows are included in the response.
    """

    dataframe = await _process_csv_upload(file)

    try:
        predictions, rejected_rows, summary = (
            _run_batch_prediction(dataframe)
        )

        prediction_preview = _dataframe_records(
            predictions,
            limit=PREVIEW_ROWS,
        )

        rejected_preview = _dataframe_records(
            rejected_rows,
            limit=PREVIEW_ROWS,
        )

        response = {
            "summary": summary,
            "preview_rows": prediction_preview,
            "preview_count": len(prediction_preview),
            "total_rows": len(predictions),
            "rejected_preview_rows": rejected_preview,
            "rejected_preview_count": len(rejected_preview),
            "total_rejected_rows": len(rejected_rows),
            "download_endpoint": (
                "/api/v1/ml/predict-csv/download"
            ),
            "rejected_download_endpoint": (
                "/api/v1/ml/predict-csv/download/rejected"
            ),
        }

        return JSONResponse(
            content=jsonable_encoder(response)
        )

    except HTTPException:
        raise

    except Exception as exc:
        _handle_batch_exception(exc)


@router.post("/predict-csv/download")
async def download_csv_predictions(
    file: UploadFile = File(...),
):
    """Download all successful batch prediction rows as CSV."""

    dataframe = await _process_csv_upload(file)

    try:
        predictions, _, _ = _run_batch_prediction(dataframe)

        output = StringIO()

        predictions.to_csv(
            output,
            index=False,
        )

        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    'attachment; filename="prediction_results.csv"'
                )
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        _handle_batch_exception(exc)


@router.post("/predict-csv/download/rejected")
async def download_csv_rejected_rows(
    file: UploadFile = File(...),
):
    """Download all validation and prediction-rejected rows as CSV."""

    dataframe = await _process_csv_upload(file)

    try:
        _, rejected_rows, _ = _run_batch_prediction(dataframe)

        output = StringIO()

        rejected_rows.to_csv(
            output,
            index=False,
        )

        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    'attachment; filename="rejected_rows.csv"'
                )
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        _handle_batch_exception(exc)
