
import hashlib
import json
import re
import shutil

from datetime import datetime, timezone
from pathlib import Path

from app.ml.datasets.dataset_validator import (
    load_dataset,
    validate_dataframe
)

from app.ml.datasets.dataset_profiler import (
    profile_dataframe,
    save_profile
)

from app.ml.datasets.schema_detector import detect_schema

from app.ml.datasets.dataset_registry import DatasetRegistry


class DatasetManager:
    """
    Dynamic dataset management system.

    Supports:
    - CSV and Excel datasets
    - Automatic validation
    - Dynamic sensor detection
    - Dataset profiling
    - SHA-256 duplicate detection
    - Custom sensor parameters
    - Custom fault types
    - Dataset registration
    - Dataset retrieval and deletion
    - Safer rollback during failed registration
    """

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

    def __init__(self, artifact_root=None, registry=None):

        backend_directory = Path(__file__).resolve().parents[3]

        self.artifact_root = Path(
            artifact_root
            if artifact_root is not None
            else backend_directory / "ml_artifacts"
        )

        self.datasets_root = self.artifact_root / "datasets"
        self.reports_root = self.artifact_root / "reports"

        self.datasets_root.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)

        registry_path = (
            self.datasets_root / "dataset_registry.json"
        )

        self.registry = registry or DatasetRegistry(
            registry_path=registry_path
        )

    @staticmethod
    def calculate_file_hash(file_path):
        """Calculate SHA-256 hash of a file."""

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024), b""
            ):
                sha256.update(chunk)

        return sha256.hexdigest()

    @staticmethod
    def _safe_name(name):
        """Convert a name into a safe folder name."""

        safe = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            str(name).strip().lower()
        ).strip("_")

        return safe or "dataset"

    def _get_dataset_directory(self, dataset_type):
        directory = (
            self.datasets_root / self._safe_name(dataset_type)
        )

        directory.mkdir(parents=True, exist_ok=True)

        return directory

    def _find_existing_hash(self, file_hash):
        """Find an already registered dataset with the same hash."""

        for record in self.registry.list_datasets():
            metadata = record.get("metadata", {})

            if metadata.get("sha256") == file_hash:
                return record

        return None

    def _rollback_registration(
        self,
        dataset_id,
        destination_path,
        profile_path
    ):
        """
        Roll back a failed registration.

        Registry removal happens before deleting stored files.
        This prevents a remaining registry record from pointing
        to a file that has already been deleted.
        """

        if dataset_id is not None:

            removed = self.registry.remove_dataset(dataset_id)

            if not removed:
                remaining_record = self.registry.get_dataset(
                    dataset_id
                )

                if remaining_record is not None:
                    raise RuntimeError(
                        "Registry rollback failed. "
                        f"Dataset record still exists: {dataset_id}"
                    )

        # Delete the profile only after registry rollback succeeds.
        if profile_path is not None and profile_path.exists():
            profile_path.unlink()

        # Delete the copied dataset only after registry rollback.
        if destination_path is not None and destination_path.exists():
            destination_path.unlink()

    def add_dataset(
        self,
        file_path,
        name=None,
        dataset_type="custom",
        task_type="classification",
        target_column=None,
        description="",
        source="local",
        exclude_columns=None,
        required_columns=None,
        min_rows=10,
        allow_duplicates=False,
        custom_parameters=None,
        metadata=None,
        **additional_parameters
    ):
        """
        Import, validate, profile and register a dataset.

        Any additional keyword arguments are automatically
        stored as custom dataset parameters.

        Example:
            vibration_sensor=True
            pressure_sensor=True
            fault_types=[...]
        """

        source_path = Path(file_path).expanduser().resolve()

        if not source_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {source_path}"
            )

        if not source_path.is_file():
            raise ValueError("Dataset path must be a file.")

        extension = source_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

        if custom_parameters is not None:
            if not isinstance(custom_parameters, dict):
                raise TypeError(
                    "custom_parameters must be a dictionary."
                )

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary.")

        if not isinstance(allow_duplicates, bool):
            raise TypeError("allow_duplicates must be a boolean.")

        if not isinstance(min_rows, int) or isinstance(min_rows, bool):
            raise TypeError("min_rows must be an integer.")

        if min_rows < 0:
            raise ValueError("min_rows cannot be negative.")

        # Merge dictionary-based and keyword-based parameters.
        # Explicit keyword arguments take precedence.
        combined_parameters = {
            **(custom_parameters or {}),
            **additional_parameters
        }

        dataset_name = name or source_path.stem

        # --------------------------------------------------
        # 1. Load dataset
        # --------------------------------------------------

        dataframe = load_dataset(source_path)

        # --------------------------------------------------
        # 2. Validate dataset
        # --------------------------------------------------

        validation = validate_dataframe(
            dataframe,
            target_column=target_column,
            required_columns=required_columns,
            min_rows=min_rows
        )

        if not validation["valid"]:
            raise ValueError({
                "message": "Dataset validation failed.",
                "errors": validation["errors"]
            })

        # --------------------------------------------------
        # 3. Calculate SHA-256
        # --------------------------------------------------

        file_hash = self.calculate_file_hash(source_path)

        # --------------------------------------------------
        # 4. Detect duplicates
        # --------------------------------------------------

        existing = self._find_existing_hash(file_hash)

        if existing and not allow_duplicates:
            return {
                "success": False,
                "duplicate": True,
                "message": "Dataset already registered.",
                "existing_dataset": existing
            }

        # --------------------------------------------------
        # 5. Detect features dynamically
        # --------------------------------------------------

        schema = detect_schema(
            dataframe,
            target_column=target_column,
            exclude_columns=exclude_columns
        )

        # --------------------------------------------------
        # 6. Generate dataset profile
        # --------------------------------------------------

        profile = profile_dataframe(
            dataframe,
            target_column=target_column
        )

        # --------------------------------------------------
        # 7. Prepare storage location
        # --------------------------------------------------

        dataset_directory = self._get_dataset_directory(
            dataset_type
        )

        safe_filename = self._safe_name(source_path.stem)

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S_%f")

        destination_path = (
            dataset_directory
            / f"{safe_filename}_{timestamp}{extension}"
        )

        profile_path = None
        registered_dataset_id = None

        # --------------------------------------------------
        # 8. Prepare metadata
        # --------------------------------------------------

        dataset_metadata = {
            **(metadata or {}),
            "sha256": file_hash,
            "original_filename": source_path.name,
            "file_format": extension,
            "file_size_bytes": source_path.stat().st_size,
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "numeric_features": schema["numeric_features"],
            "categorical_features": schema["categorical_features"],
            "datetime_columns": schema["datetime_columns"],
            "possible_target_columns": schema["possible_target_columns"],
            "excluded_columns": schema["excluded_columns"],
            "validation": validation,
            "schema": schema,
            "imported_at": datetime.now(
                timezone.utc
            ).isoformat()
        }

        # --------------------------------------------------
        # 9. Copy, register and save profile
        # --------------------------------------------------

        try:

            # Copy source dataset.
            shutil.copy2(source_path, destination_path)

            # Register dataset with custom parameters.
            record = self.registry.register_dataset(
                name=dataset_name,
                file_path=str(destination_path),
                dataset_type=dataset_type,
                task_type=task_type,
                target_column=target_column,
                description=description,
                source=source,
                metadata=dataset_metadata,
                **combined_parameters
            )

            registered_dataset_id = record["dataset_id"]

            # Save profile using generated dataset ID.
            profile_path = (
                self.reports_root
                / f"{registered_dataset_id}_profile.json"
            )

            save_profile(profile, profile_path)

            # Update registry with profile location.
            updated_metadata = {
                **record["metadata"],
                "profile_report": str(profile_path)
            }

            record = self.registry.update_dataset(
                registered_dataset_id,
                {
                    "metadata": updated_metadata,
                    "status": "validated"
                }
            )

            if record is None:
                raise RuntimeError(
                    "Dataset was registered, but registry "
                    "metadata update returned no record."
                )

        except Exception as original_error:

            try:
                self._rollback_registration(
                    dataset_id=registered_dataset_id,
                    destination_path=destination_path,
                    profile_path=profile_path
                )

            except Exception as rollback_error:

                # Preserve stored files if rollback fails.
                # This is safer than leaving a registry entry
                # pointing to a deleted dataset file.
                message = (
                    "Dataset import failed and rollback also failed. "
                    "Stored files were preserved to protect registry "
                    "consistency. "
                    f"Original error: {original_error}. "
                    f"Rollback error: {rollback_error}."
                )

                if hasattr(original_error, "add_note"):
                    original_error.add_note(message)

            raise

        # --------------------------------------------------
        # 10. Return successful import result
        # --------------------------------------------------

        return {
            "success": True,
            "duplicate": False,
            "dataset": record,
            "validation": validation,
            "schema": schema,
            "profile": profile,
            "stored_file": str(destination_path),
            "profile_report": str(profile_path)
        }

    def get_dataset(self, dataset_id):
        return self.registry.get_dataset(dataset_id)

    def list_datasets(
        self,
        dataset_type=None,
        task_type=None,
        status=None
    ):
        return self.registry.list_datasets(
            dataset_type=dataset_type,
            task_type=task_type,
            status=status
        )

    def get_dataset_dataframe(self, dataset_id):
        record = self.get_dataset(dataset_id)

        if record is None:
            raise ValueError(
                f"Unknown dataset ID: {dataset_id}"
            )

        file_path = Path(record["file_path"])

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file missing: {file_path}"
            )

        return load_dataset(file_path)

    def get_dataset_schema(self, dataset_id):
        record = self.get_dataset(dataset_id)

        if record is None:
            raise ValueError(
                f"Unknown dataset ID: {dataset_id}"
            )

        return record.get("metadata", {}).get("schema", {})

    def get_custom_parameters(self, dataset_id):
        record = self.get_dataset(dataset_id)

        if record is None:
            raise ValueError(
                f"Unknown dataset ID: {dataset_id}"
            )

        return record.get("custom_parameters", {})

    def remove_dataset(self, dataset_id, delete_file=False):
        """
        Remove a dataset registry record.

        Dataset and profile files are deleted only when
        delete_file=True.
        """

        record = self.get_dataset(dataset_id)

        if record is None:
            return False

        removed = self.registry.remove_dataset(dataset_id)

        if removed and delete_file:

            file_path = Path(record["file_path"])

            if file_path.exists():
                file_path.unlink()

            profile_path = (
                record.get("metadata", {}).get("profile_report")
            )

            if profile_path:
                profile_file = Path(profile_path)

                if profile_file.exists():
                    profile_file.unlink()

        return removed

    def export_registry(self, output_path):
        records = self.registry.list_datasets()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                records,
                file,
                indent=4,
                ensure_ascii=False
            )

        return str(path)
