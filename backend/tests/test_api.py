
import pandas as pd
import pytest

from fastapi.testclient import TestClient

from app.main import app
from app.api import ml_routes


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_batch_result():
    predictions = pd.DataFrame({
        "Type": ["L", "M"],
        "failure_probability": [0.01, 0.75],
        "risk_level": ["Low", "High"],
        "processing_status": ["SUCCESS", "SUCCESS"],
    })

    rejected = pd.DataFrame({
        "Type": ["INVALID"],
        "row_number": [3],
        "processing_status": ["REJECTED_VALIDATION"],
        "rejection_stage": ["VALIDATION"],
        "rejection_reason": [
            "Type: invalid machine type 'INVALID'; expected L, M, or H"
        ],
    })

    summary = {
        "status": "COMPLETED",
        "total_rows": 3,
        "successful_predictions": 2,
        "rejected_rows": 1,
        "validation_rejections": 1,
        "prediction_rejections": 0,
        "prediction_engine": "integrated_engine",
    }

    return {
        "predictions": predictions,
        "rejected_rows": rejected,
        "summary": summary,
    }


def test_health_endpoint(client):
    response = client.get("/api/v1/ml/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["service"] == "AI Predictive Maintenance API"
    assert body["version"] == "1.0.0"


def test_single_prediction_success(client, monkeypatch):
    expected_result = {
        "success": True,
        "equipment_id": "TEST-MACHINE-001",
        "failure_prediction": "NORMAL",
        "failure_probability": 0.001,
        "risk_level": "Low",
        "rul_available": False,
        "rul_escalation_applied": False,
    }

    received_payload = {}

    def fake_predict(payload):
        received_payload.update(payload)
        return expected_result

    monkeypatch.setattr(
        ml_routes.ml_service,
        "predict",
        fake_predict,
    )

    response = client.post(
        "/api/v1/ml/predict",
        json={
            "equipment_id": "TEST-MACHINE-001",
            "sensor_data": {
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100.0,
                "Type": "L",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["risk_level"] == "Low"

    assert received_payload["equipment_id"] == "TEST-MACHINE-001"
    assert received_payload["sensor_data"]["Type"] == "L"
    assert received_payload["rul_sensor_data"] is None


def test_single_prediction_rejects_missing_sensor_data(client):
    response = client.post(
        "/api/v1/ml/predict",
        json={
            "equipment_id": "TEST-MACHINE-002",
        },
    )

    assert response.status_code == 422


def test_single_prediction_rejects_unknown_request_fields(client):
    response = client.post(
        "/api/v1/ml/predict",
        json={
            "sensor_data": {
                "Type": "L",
            },
            "unexpected_field": "not allowed",
        },
    )

    assert response.status_code == 422


def test_single_prediction_returns_503_when_model_is_missing(
    client,
    monkeypatch,
):
    def fake_predict(payload):
        raise FileNotFoundError("Private model path")

    monkeypatch.setattr(
        ml_routes.ml_service,
        "predict",
        fake_predict,
    )

    response = client.post(
        "/api/v1/ml/predict",
        json={
            "sensor_data": {
                "Type": "L",
            },
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "A required prediction model is unavailable."
    )
    assert "Private model path" not in response.text


def test_csv_prediction_returns_predictions_and_rejections(
    client,
    monkeypatch,
    sample_batch_result,
):
    received_dataframe = {}

    def fake_batch_prediction(dataframe):
        received_dataframe["dataframe"] = dataframe.copy()
        return sample_batch_result

    monkeypatch.setattr(
        ml_routes,
        "predict_batch_dataframe",
        fake_batch_prediction,
    )

    response = client.post(
        "/api/v1/ml/predict-csv",
        files={
            "file": (
                "sample.csv",
                (
                    "Type,Rotational speed [rpm]\n"
                    "L,1500\n"
                    "M,1600\n"
                    "INVALID,1700\n"
                ),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["summary"] == sample_batch_result["summary"]
    assert body["total_rows"] == 2
    assert body["preview_count"] == 2

    assert body["total_rejected_rows"] == 1
    assert body["rejected_preview_count"] == 1

    assert body["preview_rows"][0]["risk_level"] == "Low"
    assert (
        body["rejected_preview_rows"][0]["processing_status"]
        == "REJECTED_VALIDATION"
    )

    assert body["download_endpoint"] == (
        "/api/v1/ml/predict-csv/download"
    )

    assert body["rejected_download_endpoint"] == (
        "/api/v1/ml/predict-csv/download/rejected"
    )

    assert len(received_dataframe["dataframe"]) == 3


def test_csv_prediction_handles_empty_prediction_dataframe(
    client,
    monkeypatch,
):
    empty_predictions = pd.DataFrame(
        columns=[
            "row_number",
            "processing_status",
            "failure_probability",
        ]
    )

    rejected = pd.DataFrame({
        "row_number": [1],
        "processing_status": ["REJECTED_VALIDATION"],
        "rejection_reason": ["Invalid Type"],
    })

    def fake_batch_prediction(dataframe):
        return {
            "predictions": empty_predictions,
            "rejected_rows": rejected,
            "summary": {
                "status": "NO_ROWS_PREDICTED",
                "total_rows": 1,
                "successful_predictions": 0,
                "rejected_rows": 1,
            },
        }

    monkeypatch.setattr(
        ml_routes,
        "predict_batch_dataframe",
        fake_batch_prediction,
    )

    response = client.post(
        "/api/v1/ml/predict-csv",
        files={
            "file": (
                "invalid.csv",
                "Type\nINVALID\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["total_rows"] == 0
    assert body["total_rejected_rows"] == 1
    assert body["preview_rows"] == []
    assert len(body["rejected_preview_rows"]) == 1


def test_csv_prediction_returns_503_when_model_is_missing(
    client,
    monkeypatch,
):
    def fake_batch_prediction(dataframe):
        raise FileNotFoundError("Private model path")

    monkeypatch.setattr(
        ml_routes,
        "predict_batch_dataframe",
        fake_batch_prediction,
    )

    response = client.post(
        "/api/v1/ml/predict-csv",
        files={
            "file": (
                "sample.csv",
                "Type\nL\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "A required prediction model is unavailable."
    )
    assert "Private model path" not in response.text


def test_csv_download_returns_prediction_csv(
    client,
    monkeypatch,
    sample_batch_result,
):
    monkeypatch.setattr(
        ml_routes,
        "predict_batch_dataframe",
        lambda dataframe: sample_batch_result,
    )

    response = client.post(
        "/api/v1/ml/predict-csv/download",
        files={
            "file": (
                "sample.csv",
                "Type\nL\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]
    assert "failure_probability" in response.text
    assert "risk_level" in response.text
    assert "REJECTED_VALIDATION" not in response.text


def test_csv_rejected_download_returns_rejected_rows(
    client,
    monkeypatch,
    sample_batch_result,
):
    monkeypatch.setattr(
        ml_routes,
        "predict_batch_dataframe",
        lambda dataframe: sample_batch_result,
    )

    response = client.post(
        "/api/v1/ml/predict-csv/download/rejected",
        files={
            "file": (
                "sample.csv",
                "Type\nINVALID\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]
    assert "rejection_reason" in response.text
    assert "REJECTED_VALIDATION" in response.text


def test_csv_upload_rejects_non_csv_file(client):
    response = client.post(
        "/api/v1/ml/predict-csv",
        files={
            "file": (
                "sample.txt",
                "some text",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Only .csv files are supported."
    )


def test_csv_upload_rejects_file_over_size_limit(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        ml_routes,
        "MAX_UPLOAD_BYTES",
        10,
    )

    response = client.post(
        "/api/v1/ml/predict-csv",
        files={
            "file": (
                "large.csv",
                "Type\n" + ("L" * 100),
                "text/csv",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "File exceeds the 25 MB upload limit."
    )
