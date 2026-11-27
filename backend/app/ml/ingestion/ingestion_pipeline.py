
"""Unified ingestion, validation, compatibility and reference evaluation."""

from pathlib import Path

from app.ml.ingestion.file_loader import load_machine_data
from app.ml.ingestion.column_mapper import map_sensor_columns
from app.ml.ingestion.sensor_validator import validate_sensor_data

from app.ml.datasets.compatibility_checker import (
    check_dataset_compatibility,
)

from app.ml.ingestion.reference_range_manager import (
    load_reference_ranges,
    evaluate_sensor_readings,
    DEFAULT_OUTPUT_PATH,
)


def _empty_validation_report(error=None):
    """Create a consistent validation report for an unsuccessful stage."""
    errors = [str(error)] if error else []

    return {
        "is_valid": False,
        "status": "FAILED",
        "errors": errors,
        "warnings": [],
    }


def _empty_compatibility_report(error=None):
    """Create a consistent compatibility report for an unsuccessful stage."""
    errors = [str(error)] if error else []

    return {
        "compatible": False,
        "errors": errors,
        "warnings": [],
        "required_features": [],
        "missing_features": [],
        "extra_columns": [],
        "leakage_columns": [],
        "numeric_checks": {},
        "categorical_checks": {},
        "missing_value_counts": {},
        "model_name": None,
    }


def _failed_result(
    file_path,
    stage,
    error,
    raw_dataframe=None,
    mapped_dataframe=None,
    mapping_result=None,
    validation_report=None,
    compatibility_report=None,
    reference_result=None,
    reference_error=None,
):
    """Return a consistent structured report for a failed ingestion stage."""
    errors = [f"{stage}: {type(error).__name__}: {error}"]

    raw_shape = (
        raw_dataframe.shape
        if raw_dataframe is not None
        else None
    )

    mapped_shape = (
        mapped_dataframe.shape
        if mapped_dataframe is not None
        else None
    )

    mapping_result = mapping_result or {}

    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "status": "REJECTED",
        "is_valid": False,
        "failed_stage": stage,
        "errors": errors,
        "raw_shape": raw_shape,
        "mapped_shape": mapped_shape,
        "column_mapping": mapping_result.get("mapping", {}),
        "unmatched_columns": mapping_result.get(
            "unmatched_columns", []
        ),
        "missing_features": mapping_result.get(
            "missing_features", []
        ),
        "validation_report": (
            validation_report
            if validation_report is not None
            else _empty_validation_report()
        ),
        "compatibility_report": (
            compatibility_report
            if compatibility_report is not None
            else _empty_compatibility_report()
        ),
        "reference_range_report": reference_result,
        "reference_range_error": reference_error,
        "warnings": [],
        "dataframe": mapped_dataframe,
    }


def ingest_and_validate(file_path, required_features=None):
    """
    Complete sensor data ingestion workflow.

    Stages:
    1. Load uploaded file.
    2. Map incoming columns.
    3. Validate sensor readings.
    4. Check compatibility with registered XGBoost model.
    5. Evaluate statistical sensor reference ranges.

    Expected input and processing errors are returned as structured
    rejection reports rather than terminating the calling workflow.

    Reference-range warnings do not automatically reject
    otherwise valid sensor data.

    Does not modify registered models or thresholds.
    """

    file_path = Path(file_path)

    if not file_path.is_file():
        error = FileNotFoundError(
            f"Uploaded file does not exist: {file_path}"
        )

        return _failed_result(
            file_path=file_path,
            stage="FILE_LOADING",
            error=error,
        )

    raw_dataframe = None
    mapped_dataframe = None
    mapping_result = None
    validation_report = None
    compatibility_result = None

    # --------------------------------------------------
    # STAGE 1: LOAD FILE
    # --------------------------------------------------

    try:
        raw_dataframe = load_machine_data(str(file_path))

        if raw_dataframe is None:
            raise ValueError("File loader returned no data.")

        if raw_dataframe.empty:
            raise ValueError("Uploaded file contains no data rows.")

    except Exception as exc:
        return _failed_result(
            file_path=file_path,
            stage="FILE_LOADING",
            error=exc,
        )

    # --------------------------------------------------
    # STAGE 2: COLUMN MAPPING
    # --------------------------------------------------

    try:
        mapping_result = map_sensor_columns(
            raw_dataframe,
            required_features=required_features,
        )

        mapped_dataframe = mapping_result["dataframe"]

        if mapped_dataframe is None:
            raise ValueError(
                "Column mapper returned no DataFrame."
            )

        if not mapping_result.get("is_valid", False):
            return _failed_result(
                file_path=file_path,
                stage="COLUMN_MAPPING",
                error=ValueError(
                    "Column mapping did not satisfy the required schema."
                ),
                raw_dataframe=raw_dataframe,
                mapped_dataframe=mapped_dataframe,
                mapping_result=mapping_result,
            )

    except Exception as exc:
        return _failed_result(
            file_path=file_path,
            stage="COLUMN_MAPPING",
            error=exc,
            raw_dataframe=raw_dataframe,
        )

    # --------------------------------------------------
    # STAGE 3: SENSOR VALIDATION
    # --------------------------------------------------

    try:
        validation_result = validate_sensor_data(
            mapped_dataframe
        )

        if not isinstance(validation_result, dict):
            raise TypeError(
                "Sensor validator did not return a dictionary."
            )

        validation_report = validation_result.get("report")

        if not isinstance(validation_report, dict):
            raise ValueError(
                "Sensor validator did not return a valid report."
            )

        validated_dataframe = validation_result.get("dataframe")

        if validated_dataframe is None:
            raise ValueError(
                "Sensor validator returned no DataFrame."
            )

        if not validation_report.get("is_valid", False):
            return _failed_result(
                file_path=file_path,
                stage="SENSOR_VALIDATION",
                error=ValueError(
                    "Sensor validation rejected the uploaded data."
                ),
                raw_dataframe=raw_dataframe,
                mapped_dataframe=validated_dataframe,
                mapping_result=mapping_result,
                validation_report=validation_report,
            )

    except Exception as exc:
        return _failed_result(
            file_path=file_path,
            stage="SENSOR_VALIDATION",
            error=exc,
            raw_dataframe=raw_dataframe,
            mapped_dataframe=mapped_dataframe,
            mapping_result=mapping_result,
            validation_report=validation_report,
        )

    # --------------------------------------------------
    # STAGE 4: MODEL COMPATIBILITY
    # --------------------------------------------------

    try:
        compatibility_result = check_dataset_compatibility(
            validated_dataframe
        )

        if not isinstance(compatibility_result, dict):
            raise TypeError(
                "Compatibility checker did not return a dictionary."
            )

        if not compatibility_result.get("compatible", False):
            return _failed_result(
                file_path=file_path,
                stage="MODEL_COMPATIBILITY",
                error=ValueError(
                    "Uploaded data is not compatible with "
                    "the registered prediction model."
                ),
                raw_dataframe=raw_dataframe,
                mapped_dataframe=validated_dataframe,
                mapping_result=mapping_result,
                validation_report=validation_report,
                compatibility_report=compatibility_result,
            )

    except Exception as exc:
        return _failed_result(
            file_path=file_path,
            stage="MODEL_COMPATIBILITY",
            error=exc,
            raw_dataframe=raw_dataframe,
            mapped_dataframe=validated_dataframe,
            mapping_result=mapping_result,
            validation_report=validation_report,
        )

    # --------------------------------------------------
    # STAGE 5: REFERENCE RANGE EVALUATION
    # --------------------------------------------------

    reference_result = None
    reference_error = None

    try:
        reference_data = load_reference_ranges(
            DEFAULT_OUTPUT_PATH
        )

        reference_result = evaluate_sensor_readings(
            validated_dataframe,
            reference_data,
        )

    except Exception as exc:
        # Reference ranges are an additional assessment.
        # Their unavailability does not invalidate model-compatible data.
        reference_error = (
            f"{type(exc).__name__}: {exc}"
        )

    # --------------------------------------------------
    # STAGE 6: FINAL INGESTION DECISION
    # --------------------------------------------------

    is_valid = (
        mapping_result.get("is_valid", False)
        and validation_report.get("is_valid", False)
        and compatibility_result.get("compatible", False)
    )

    status = (
        "READY_FOR_PREDICTION"
        if is_valid
        else "REJECTED"
    )

    warnings = []

    warnings.extend(
        validation_report.get("warnings", [])
    )

    warnings.extend(
        compatibility_result.get("warnings", [])
    )

    if reference_result:
        warnings.extend(
            reference_result.get("warnings", [])
        )

    if reference_error:
        warnings.append(
            "Reference range evaluation unavailable: "
            + reference_error
        )

    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "status": status,
        "is_valid": is_valid,
        "failed_stage": None,
        "errors": [],
        "raw_shape": raw_dataframe.shape,
        "mapped_shape": validated_dataframe.shape,
        "column_mapping": mapping_result.get("mapping", {}),
        "unmatched_columns": mapping_result.get(
            "unmatched_columns", []
        ),
        "missing_features": mapping_result.get(
            "missing_features", []
        ),
        "validation_report": validation_report,
        "compatibility_report": compatibility_result,
        "reference_range_report": reference_result,
        "reference_range_error": reference_error,
        "warnings": warnings,
        "dataframe": validated_dataframe,
    }


def print_ingestion_report(result):
    """Print a readable report of the complete ingestion process."""

    print("\n" + "=" * 65)
    print("PREDICTIVE MAINTENANCE - INGESTION REPORT")
    print("=" * 65)

    print("File:", result.get("file_name"))
    print("Raw shape:", result.get("raw_shape"))
    print("Mapped shape:", result.get("mapped_shape"))

    if result.get("failed_stage"):
        print("Failed stage:", result["failed_stage"])

    print("\n1. COLUMN MAPPING")

    for original, standard in result.get(
        "column_mapping", {}
    ).items():
        print(f"{original} -> {standard or 'Unmatched'}")

    print("Unmatched columns:", result.get("unmatched_columns", []))
    print("Missing features:", result.get("missing_features", []))

    print("\n2. SENSOR VALIDATION")

    validation = result.get("validation_report") or {}

    print("Status:", validation.get("status", "NOT RUN"))
    print("Errors:", validation.get("errors", []))

    print("\n3. MODEL COMPATIBILITY")

    compatibility = result.get("compatibility_report") or {}

    print("Compatible:", compatibility.get("compatible", False))
    print("Model:", compatibility.get("model_name"))
    print("Errors:", compatibility.get("errors", []))

    print("\n4. REFERENCE RANGE EVALUATION")

    reference = result.get("reference_range_report")

    if reference:
        for sensor, details in reference.get(
            "sensor_results", {}
        ).items():
            print(
                f"{sensor}: "
                f"{details.get('outside_reference_count', 'N/A')} "
                "readings outside statistical reference range"
            )
    else:
        print(
            "Unavailable:",
            result.get("reference_range_error") or "Not evaluated"
        )

    print("\n5. ERRORS")

    for error in result.get("errors", []):
        print("-", error)

    print("\n6. WARNINGS")

    for warning in result.get("warnings", []):
        print("-", warning)

    print("\nFINAL INGESTION STATUS:", result.get("status"))
    print("=" * 65)


if __name__ == "__main__":

    test_file = Path(
        "ml_artifacts/monitoring/ingestion_test.csv"
    )

    result = ingest_and_validate(test_file)

    print_ingestion_report(result)

    if result["is_valid"]:
        print("\nINGESTION PIPELINE SELF-TEST PASSED")
    else:
        print("\nINGESTION PIPELINE SELF-TEST FAILED")
