
"""Model registry with cached model loading and fresh metadata reads."""

from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
import json
import re

import joblib
import sklearn


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[3]
REGISTRY_DIR = BASE_DIR / "ml_artifacts" / "model_registry"

REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# MODEL NAME VALIDATION
# ==========================================

def _safe_model_name(model_name):
    """Convert model name into a safe folder name."""

    name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        model_name.strip().lower()
    ).strip("_")

    if not name:
        raise ValueError("Model name cannot be empty.")

    return name


# ==========================================
# CACHE INVALIDATION
# ==========================================

def _clear_explanation_cache():
    """
    Clear cached SHAP explainers.

    Import locally to avoid a circular import:
    prediction_explainer imports this model registry.
    """

    try:
        from app.ml.explainability.prediction_explainer import (
            clear_explanation_cache
        )

        clear_explanation_cache()

    except ImportError:
        # SHAP explanation module may not be available
        # in environments that only use model registry.
        pass


def clear_model_cache():
    """Clear all cached fitted model objects."""

    _load_cached_model.cache_clear()


def clear_all_model_caches():
    """Clear both fitted-model and SHAP explanation caches."""

    clear_model_cache()
    _clear_explanation_cache()


# ==========================================
# CACHED MODEL FILE LOADING
# ==========================================

@lru_cache(maxsize=32)
def _load_cached_model(model_path, modified_ns, file_size):
    """
    Load and cache a fitted model.

    Modification time and file size are part of the cache key.
    Replacing the model file creates a fresh cache entry.
    """

    return joblib.load(model_path)


# ==========================================
# SAVE MODEL
# ==========================================

def save_model_bundle(
    model_name,
    model,
    threshold,
    model_type,
    feature_names,
    metrics=None,
    best_parameters=None,
    description=""
):
    """Save the complete fitted model pipeline and metadata."""

    safe_name = _safe_model_name(model_name)

    model_dir = REGISTRY_DIR / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"

    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "threshold": float(threshold),
        "feature_names": list(feature_names),
        "metrics": metrics or {},
        "best_parameters": best_parameters or {},
        "description": description,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "sklearn_version": sklearn.__version__,
        "artifact_format": "joblib",
        "model_file": "model.joblib"
    }

    temporary_model = model_dir / "model.tmp"
    temporary_metadata = model_dir / "metadata.tmp"

    joblib.dump(model, temporary_model)

    with open(
        temporary_metadata,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            default=str
        )

    temporary_model.replace(model_path)
    temporary_metadata.replace(metadata_path)

    # Invalidate both caches after successful registration.
    clear_all_model_caches()

    print("\nMODEL REGISTERED")
    print("Name:", model_name)
    print("Model:", model_path)
    print("Metadata:", metadata_path)
    print("Model and SHAP caches cleared.")

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metadata": metadata
    }


# ==========================================
# LOAD MODEL
# ==========================================

def load_model_bundle(model_name):
    """
    Load a fitted model and its latest metadata.

    Model objects are cached using artifact version information.
    Metadata is read from disk on every call.
    """

    safe_name = _safe_model_name(model_name)

    model_dir = REGISTRY_DIR / safe_name

    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}"
        )

    model_stat = model_path.stat()

    model = _load_cached_model(
        str(model_path.resolve()),
        model_stat.st_mtime_ns,
        model_stat.st_size
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    print("\nMODEL READY:", model_name)

    return {
        "model": model,
        "metadata": metadata
    }


# ==========================================
# LIST REGISTERED MODELS
# ==========================================

def list_registered_models():
    """Return metadata for every registered model."""

    registered = []

    if not REGISTRY_DIR.exists():
        return registered

    for model_dir in REGISTRY_DIR.iterdir():

        if not model_dir.is_dir():
            continue

        metadata_path = model_dir / "metadata.json"

        if not metadata_path.exists():
            continue

        try:
            with open(
                metadata_path,
                "r",
                encoding="utf-8"
            ) as file:
                metadata = json.load(file)

            registered.append(metadata)

        except (OSError, json.JSONDecodeError) as error:
            print(
                f"Could not read metadata for "
                f"{model_dir.name}: {error}"
            )

    return registered


# ==========================================
# DELETE REGISTERED MODEL
# ==========================================

def delete_model_bundle(model_name):
    """Delete one registered model after explicit invocation."""

    safe_name = _safe_model_name(model_name)

    model_dir = REGISTRY_DIR / safe_name

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model registry entry not found: {model_name}"
        )

    for file_path in model_dir.iterdir():
        if file_path.is_file():
            file_path.unlink()

    model_dir.rmdir()

    # Invalidate both caches after successful deletion.
    clear_all_model_caches()

    print("MODEL DELETED:", model_name)
    print("Model and SHAP caches cleared.")
