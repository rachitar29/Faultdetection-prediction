"""
HTTP client for the AI Predictive Maintenance dashboard.
All ML processing remains in the FastAPI backend.
"""

import requests

DEFAULT_API_URL = "http://127.0.0.1:8001"
REQUEST_TIMEOUT = 180


class APIClientError(Exception):
    """Raised when the dashboard cannot complete an API request."""


class PredictiveMaintenanceAPI:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, endpoint: str, **kwargs):
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault("timeout", REQUEST_TIMEOUT)

        try:
            response = requests.request(
                method=method,
                url=url,
                **kwargs,
            )
        except requests.exceptions.ConnectionError as exc:
            raise APIClientError(
                "Cannot connect to the FastAPI backend. "
                "Check that the backend is running."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise APIClientError(
                "The backend request timed out. "
                "Try again or use a smaller CSV."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise APIClientError(
                f"Backend communication failed: {exc}"
            ) from exc

        if not response.ok:
            try:
                error_data = response.json()
                detail = error_data.get("detail", response.text)
            except ValueError:
                detail = response.text

            raise APIClientError(
                f"API error ({response.status_code}): {detail}"
            )

        return response

    def health(self):
        response = self._request(
            "GET",
            "/api/v1/ml/health",
            timeout=10,
        )
        return response.json()

    def predict_single(self, payload):
        response = self._request(
            "POST",
            "/api/v1/ml/predict",
            json=payload,
        )
        return response.json()

    def predict_csv(self, filename, file_bytes):
        response = self._request(
            "POST",
            "/api/v1/ml/predict-csv",
            files={
                "file": (
                    filename,
                    file_bytes,
                    "text/csv",
                )
            },
        )
        return response.json()

    def download_predictions(self, filename, file_bytes):
        response = self._request(
            "POST",
            "/api/v1/ml/predict-csv/download",
            files={
                "file": (
                    filename,
                    file_bytes,
                    "text/csv",
                )
            },
        )
        return response.content

    def download_rejected_rows(self, filename, file_bytes):
        response = self._request(
            "POST",
            "/api/v1/ml/predict-csv/download/rejected",
            files={
                "file": (
                    filename,
                    file_bytes,
                    "text/csv",
                )
            },
        )
        return response.content
