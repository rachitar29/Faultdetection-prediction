
import json
import re
import uuid

from datetime import datetime, timezone
from pathlib import Path


def get_default_registry_path():
    """Return the default dataset registry location."""

    backend_dir = Path(__file__).resolve().parents[3]

    return (
        backend_dir
        / "ml_artifacts"
        / "datasets"
        / "dataset_registry.json"
    )


class DatasetRegistry:
    """
    Maintain a JSON registry of imported datasets.

    Supports:
    - Multiple dataset types
    - Classification, regression, RUL and anomaly detection
    - Custom sensor parameters
    - Custom fault types
    - Dataset metadata
    - Dataset registration and updates
    """

    SUPPORTED_TASK_TYPES = {
        "classification",
        "regression",
        "anomaly_detection",
        "rul",
        "clustering",
        "unknown"
    }

    def __init__(self, registry_path=None):

        self.registry_path = Path(
            registry_path or get_default_registry_path()
        )

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not self.registry_path.exists():
            self._save({
                "registry_version": 2,
                "datasets": []
            })

    def _load(self):
        """Load registry from JSON."""

        try:
            with self.registry_path.open(
                "r",
                encoding="utf-8"
            ) as file:
                data = json.load(file)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Registry JSON is corrupted: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError("Invalid registry structure.")

        if not isinstance(data.get("datasets"), list):
            raise ValueError(
                "Registry must contain a datasets list."
            )

        # Maintain compatibility with older registry files.
        data.setdefault("registry_version", 2)

        for record in data["datasets"]:
            record.setdefault("custom_parameters", {})
            record.setdefault("metadata", {})

        return data

    def _save(self, data):
        """Save registry using a temporary file."""

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary_path = self.registry_path.with_suffix(".tmp")

        with temporary_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )

        temporary_path.replace(self.registry_path)

    @staticmethod
    def _generate_dataset_id(name):
        """Generate a unique dataset ID."""

        slug = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            name.lower()
        ).strip("_")

        unique_part = uuid.uuid4().hex[:8]

        return f"{slug or 'dataset'}_{unique_part}"

    def register_dataset(
        self,
        name,
        file_path,
        dataset_type="custom",
        task_type="classification",
        target_column=None,
        description="",
        source="local",
        metadata=None,
        dataset_id=None,
        **custom_parameters
    ):
        """
        Register a dataset with flexible custom parameters.

        Custom parameters can include:
        - Sensor availability
        - Sensor names
        - Fault types
        - Operating conditions
        - Machine information
        - Custom dataset-specific configuration
        """

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name is required.")

        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("Dataset file path is required.")

        if task_type not in self.SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"Unsupported task type: {task_type}"
            )

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary.")

        data = self._load()

        new_id = (
            dataset_id.strip()
            if isinstance(dataset_id, str) and dataset_id.strip()
            else self._generate_dataset_id(name)
        )

        if any(
            item.get("dataset_id") == new_id
            for item in data["datasets"]
        ):
            raise ValueError(
                f"Dataset ID already exists: {new_id}"
            )

        now = datetime.now(timezone.utc).isoformat()

        record = {
            "dataset_id": new_id,
            "name": name.strip(),
            "file_path": str(Path(file_path)),
            "dataset_type": dataset_type,
            "task_type": task_type,
            "target_column": target_column,
            "description": description,
            "source": source,
            "metadata": metadata or {},
            "custom_parameters": custom_parameters,
            "registered_at": now,
            "updated_at": now,
            "status": "registered"
        }

        data["datasets"].append(record)

        self._save(data)

        return record

    def get_dataset(self, dataset_id):
        """Retrieve one dataset using its ID."""

        data = self._load()

        for record in data["datasets"]:
            if record.get("dataset_id") == dataset_id:
                return record

        return None

    def list_datasets(
        self,
        dataset_type=None,
        task_type=None,
        status=None
    ):
        """List datasets with optional filters."""

        data = self._load()

        records = data["datasets"]

        if dataset_type is not None:
            records = [
                item for item in records
                if item.get("dataset_type") == dataset_type
            ]

        if task_type is not None:
            records = [
                item for item in records
                if item.get("task_type") == task_type
            ]

        if status is not None:
            records = [
                item for item in records
                if item.get("status") == status
            ]

        return records

    def update_dataset(self, dataset_id, updates):
        """Update dataset configuration and metadata."""

        if not isinstance(updates, dict):
            raise TypeError("Updates must be a dictionary.")

        protected_fields = {
            "dataset_id",
            "registered_at"
        }

        if protected_fields.intersection(updates):
            raise ValueError(
                "Dataset ID and registration date cannot be changed."
            )

        allowed_fields = {
            "name",
            "file_path",
            "dataset_type",
            "task_type",
            "target_column",
            "description",
            "source",
            "metadata",
            "custom_parameters",
            "status"
        }

        invalid_fields = set(updates) - allowed_fields

        if invalid_fields:
            raise ValueError(
                f"Unsupported update fields: {sorted(invalid_fields)}"
            )

        if "task_type" in updates:
            if updates["task_type"] not in self.SUPPORTED_TASK_TYPES:
                raise ValueError("Invalid task type.")

        if "metadata" in updates and not isinstance(
            updates["metadata"], dict
        ):
            raise TypeError("Metadata must be a dictionary.")

        if "custom_parameters" in updates and not isinstance(
            updates["custom_parameters"], dict
        ):
            raise TypeError(
                "Custom parameters must be a dictionary."
            )

        data = self._load()

        for record in data["datasets"]:

            if record.get("dataset_id") == dataset_id:

                record.update(updates)

                record["updated_at"] = (
                    datetime.now(timezone.utc).isoformat()
                )

                self._save(data)

                return record

        return None

    def update_custom_parameters(
        self,
        dataset_id,
        parameters
    ):
        """Add or update custom dataset parameters."""

        if not isinstance(parameters, dict):
            raise TypeError(
                "Parameters must be supplied as a dictionary."
            )

        record = self.get_dataset(dataset_id)

        if record is None:
            return None

        current_parameters = record.get(
            "custom_parameters", {}
        )

        current_parameters.update(parameters)

        return self.update_dataset(
            dataset_id,
            {
                "custom_parameters": current_parameters
            }
        )

    def remove_dataset(self, dataset_id):
        """
        Remove only the registry record.

        The original dataset file is never deleted.
        """

        data = self._load()

        original_count = len(data["datasets"])

        data["datasets"] = [
            item for item in data["datasets"]
            if item.get("dataset_id") != dataset_id
        ]

        if len(data["datasets"]) == original_count:
            return False

        self._save(data)

        return True

    def count(self):
        """Return total registered datasets."""

        return len(self._load()["datasets"])
