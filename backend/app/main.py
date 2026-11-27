import logging
import time

from fastapi import FastAPI, Request

from app.api.ml_routes import router as ml_router


logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="AI Predictive Maintenance API",
    description=(
        "Machine failure prediction, anomaly detection, "
        "and predictive maintenance assessment."
    ),
    version="1.0.0",
)


@app.middleware("http")
async def log_api_requests(request: Request, call_next):
    method = request.method
    path = request.url.path
    client = request.client.host if request.client else "unknown"
    start_time = time.perf_counter()

    logger.info(
        "[API REQUEST] %s %s | Client: %s",
        method, path, client
    )

    try:
        response = await call_next(request)
        duration = time.perf_counter() - start_time

        logger.info(
            "[API RESPONSE] %s %s | Status: %s | Duration: %.3fs",
            method, path, response.status_code, duration
        )

        return response

    except Exception:
        duration = time.perf_counter() - start_time

        logger.exception(
            "[API ERROR] %s %s | Duration: %.3fs",
            method, path, duration
        )
        raise


app.include_router(ml_router)


@app.get("/")
def root():
    return {
        "message": "AI Predictive Maintenance API is running",
        "status": "healthy",
        "docs": "/docs",
        "ml_endpoints": "/api/v1/ml",
    }
