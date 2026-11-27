
import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DynamicDatasetPreprocessor:
    """
    Reusable preprocessing engine for multiple industrial datasets.

    Supports:
    - Automatic numeric and categorical feature detection
    - JSON configuration
    - Identifier and leakage-column exclusion
    - Missing-value imputation
    - Categorical encoding
    - Optional numeric scaling
    - Reusable fitted preprocessing
    - Save and load fitted artifacts
    """

    def __init__(self, config: Any):
        if isinstance(config, (str, Path)):
            config_path = Path(config)

            if not config_path.exists():
                raise FileNotFoundError(
                    f"Configuration not found: {config_path}"
                )

            with open(config_path, "r", encoding="utf-8") as file:
                config = json.load(file)

        if not isinstance(config, dict):
            raise ValueError(
                "Configuration must be a dictionary or JSON path."
            )

        self.config = config
        self.target_column = self._get_target_column()

        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.feature_columns: list[str] = []
        self.output_feature_names: list[str] = []

        self.preprocessor: Optional[ColumnTransformer] = None
        self.is_fitted = False

        preprocessing_config = config.get("preprocessing", {})

        self.scale_numeric = preprocessing_config.get(
            "scale_numeric", True
        )
        self.numeric_strategy = preprocessing_config.get(
            "numeric_strategy", "median"
        )
        self.categorical_strategy = preprocessing_config.get(
            "categorical_strategy", "most_frequent"
        )

    def _get_target_column(self) -> Optional[str]:
        target = self.config.get("target")

        if isinstance(target, dict):
            return target.get("column")

        if isinstance(target, str):
            return target

        return self.config.get("target_column")

    def _get_configured_columns(self, keys: list[str]) -> set[str]:
        columns: set[str] = set()

        for key in keys:
            values = self.config.get(key, [])

            if isinstance(values, list):
                columns.update(values)

        return columns

    def _get_excluded_columns(self) -> set[str]:
        excluded = self._get_configured_columns([
            "identifier_columns",
            "id_columns",
            "exclude_columns",
            "excluded_columns",
            "columns_to_exclude",
            "failure_mode_columns"
        ])

        if self.target_column:
            excluded.add(self.target_column)

        timestamp_config = self.config.get("timestamp", {})

        if isinstance(timestamp_config, dict):
            timestamp_column = timestamp_config.get("column")

            if timestamp_column:
                excluded.add(timestamp_column)

        return excluded

    def _get_configured_feature_types(
        self
    ) -> tuple[set[str], set[str]]:

        numeric = set(self.config.get("numeric_features", []))
        categorical = set(
            self.config.get("categorical_features", [])
        )

        sensors = self.config.get("sensors", {})

        for column, settings in sensors.items():
            if isinstance(settings, dict):
                if settings.get("enabled", True):
                    numeric.add(column)

        categorical_rules = self.config.get(
            "categorical_rules", {}
        )
        categorical.update(categorical_rules.keys())

        return numeric, categorical

    def detect_features(self, df: pd.DataFrame) -> dict:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Cannot preprocess an empty dataset."
            )

        if df.columns.duplicated().any():
            raise ValueError(
                "Dataset contains duplicate column names."
            )

        excluded = self._get_excluded_columns()

        configured_numeric, configured_categorical = (
            self._get_configured_feature_types()
        )

        numeric: list[str] = []
        categorical: list[str] = []

        for column in df.columns:
            if column in excluded:
                continue

            if column in configured_numeric:
                numeric.append(column)
                continue

            if column in configured_categorical:
                categorical.append(column)
                continue

            if pd.api.types.is_numeric_dtype(df[column]):
                numeric.append(column)

            elif (
                pd.api.types.is_object_dtype(df[column])
                or pd.api.types.is_string_dtype(df[column])
                or isinstance(df[column].dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(df[column])
            ):
                categorical.append(column)

        missing_configured = (
            configured_numeric | configured_categorical
        ) - set(df.columns)

        if missing_configured:
            raise ValueError(
                "Configured feature columns missing from dataset: "
                f"{sorted(missing_configured)}"
            )

        overlap = set(numeric) & set(categorical)

        if overlap:
            raise ValueError(
                f"Features cannot be both numeric and categorical: {overlap}"
            )

        if not numeric and not categorical:
            raise ValueError(
                "No usable input features were detected."
            )

        self.numeric_columns = numeric
        self.categorical_columns = categorical
        self.feature_columns = numeric + categorical

        return {
            "numeric_features": numeric,
            "categorical_features": categorical,
            "excluded_columns": sorted(excluded),
            "all_features": self.feature_columns
        }

    def _build_preprocessor(self) -> None:
        numeric_steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(strategy=self.numeric_strategy)
            )
        ]

        if self.scale_numeric:
            numeric_steps.append(
                ("scaler", StandardScaler())
            )

        numeric_pipeline = Pipeline(steps=numeric_steps)

        categorical_steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy=self.categorical_strategy
                )
            ),
            (
                "encoder",
                self._create_encoder()
            )
        ]

        categorical_pipeline = Pipeline(
            steps=categorical_steps
        )

        transformers: list[tuple[str, Any, list[str]]] = []

        if self.numeric_columns:
            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    self.numeric_columns
                )
            )

        if self.categorical_columns:
            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_columns
                )
            )

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.0
        )

    @staticmethod
    def _create_encoder() -> OneHotEncoder:
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        except TypeError:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [
            column
            for column in self.feature_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Required preprocessing features missing: {missing}"
            )

        return df[self.feature_columns].copy()

    def _prepare_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare raw features for a model Pipeline.
        The model Pipeline performs its own preprocessing.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted. Call fit() first."
            )

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Model input must be a pandas DataFrame."
            )

        X = self._select_features(df)

        for column in self.numeric_columns:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

        for column in self.categorical_columns:
            X[column] = X[column].apply(
                lambda value: (
                    str(value).strip()
                    if pd.notna(value)
                    else np.nan
                )
            )

        return X

    def fit(self, df: pd.DataFrame) -> dict:
        detected = self.detect_features(df)

        X = self._select_features(df)

        if self.target_column and self.target_column in X.columns:
            raise ValueError(
                "Target column must not be used as a feature."
            )

        for column in self.numeric_columns:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

        for column in self.categorical_columns:
            X[column] = X[column].apply(
                lambda value: (
                    str(value).strip()
                    if pd.notna(value)
                    else np.nan
                )
            )

        self._build_preprocessor()

        preprocessor = self.preprocessor

        if preprocessor is None:
            raise RuntimeError(
                "Failed to build preprocessing pipeline."
            )

        preprocessor.fit(X)

        self.output_feature_names = (
            preprocessor.get_feature_names_out().tolist()
        )

        self.is_fitted = True

        return detected

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted. Call fit() first."
            )

        X = self._prepare_input(df)

        preprocessor = self.preprocessor

        if preprocessor is None:
            raise RuntimeError(
                "Preprocessing pipeline is unavailable."
            )

        transformed = preprocessor.transform(X)

        if sparse.issparse(transformed):
            transformed = transformed.toarray()

        transformed_array = np.asarray(transformed)

        return pd.DataFrame(
            transformed_array,
            columns=self.output_feature_names,
            index=df.index
        )

    def fit_transform(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:

        self.fit(df)

        X = self.transform(df)

        y = None

        if (
            self.target_column
            and self.target_column in df.columns
        ):
            y = df[self.target_column].copy()

        return X, y

    def get_feature_information(self) -> dict:
        return {
            "target_column": self.target_column,
            "numeric_features": self.numeric_columns,
            "categorical_features": self.categorical_columns,
            "feature_count_before_encoding": len(
                self.feature_columns
            ),
            "feature_count_after_encoding": len(
                self.output_feature_names
            ),
            "output_features": self.output_feature_names
        }

    def save(self, output_path: str | Path) -> Path:
        if not self.is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted preprocessor."
            )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "config": self.config,
            "target_column": self.target_column,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "feature_columns": self.feature_columns,
            "output_feature_names": self.output_feature_names,
            "preprocessor": self.preprocessor,
            "is_fitted": self.is_fitted
        }

        joblib.dump(artifact, path)

        return path

    @classmethod
    def load(
        cls,
        artifact_path: str | Path
    ) -> "DynamicDatasetPreprocessor":

        path = Path(artifact_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessing artifact not found: {path}"
            )

        artifact = joblib.load(path)

        instance = cls(artifact["config"])

        instance.target_column = artifact["target_column"]
        instance.numeric_columns = artifact["numeric_columns"]
        instance.categorical_columns = artifact["categorical_columns"]
        instance.feature_columns = artifact["feature_columns"]
        instance.output_feature_names = artifact[
            "output_feature_names"
        ]
        instance.preprocessor = artifact["preprocessor"]
        instance.is_fitted = artifact["is_fitted"]

        return instance
