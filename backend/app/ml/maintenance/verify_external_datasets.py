
"""
Generic External Multi-Dataset Verification and Evaluation.

- Inspects external CSV, TXT, XLSX and XLS datasets.
- Checks AI4I and NASA C-MAPSS schema compatibility.
- Detects common labelled classification and regression targets.
- Independently evaluates Random Forest, Extra Trees and XGBoost.
- Reports confusion matrices and classification metrics.
- Does not modify, retrain or register existing project models.

Evaluation measures agreement with supplied dataset labels.
It does not independently establish real-world ground truth.
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_DIR = BASE_DIR / "ml_data" / "external_datasets"
REPORT_DIR = BASE_DIR / "ml_artifacts" / "reports"

REPORT_FILE = REPORT_DIR / "external_dataset_verification.json"

RANDOM_STATE = 42
TEST_SIZE = 0.25

AI4I_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]

NASA_FEATURES = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

NASA_COLUMNS = ["unit", "cycle"] + NASA_FEATURES

# Common target names, independent of any particular company dataset.
KNOWN_TARGETS = {
    "machine failure",
    "machine_failure",
    "failure",
    "target",
    "label",
    "class",
    "rul",
    "remaining useful life",
    "remaining_useful_life",
    "remainingusefullife",
}

# General metadata and identifier columns that should not be model inputs.
EXCLUDED_FEATURE_TOKENS = [
    "record_id",
    "equipment_id",
    "machine_id",
    "serial_number",
    "timestamp",
    "raw_payload",
    "payload_note",
    "source_endpoint",
    "destination_endpoint",
]


def load_dataset(path):
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)

    elif suffix == ".txt":
        df = pd.read_csv(path, sep=r"\s+", header=None)

    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Recognise headerless NASA C-MAPSS files.
    if (
        len(df.columns) == 26
        and all(isinstance(column, int) for column in df.columns)
    ):
        df.columns = NASA_COLUMNS

    return df


def check_numeric_quality(df):
    numeric_columns = df.select_dtypes(include="number")

    if numeric_columns.empty:
        return 0, 0

    values = numeric_columns.to_numpy(dtype=float)

    return (
        len(numeric_columns.columns),
        int((~np.isfinite(values)).sum()),
    )


def inspect_schema(df):
    columns = [str(column) for column in df.columns]

    ai4i_missing = [
        feature for feature in AI4I_FEATURES
        if feature not in columns
    ]

    nasa_missing = [
        feature for feature in NASA_FEATURES
        if feature not in columns
    ]

    compatible_models = []

    if not ai4i_missing:
        compatible_models.append("AI4I feature schema")

    if not nasa_missing:
        compatible_models.append("NASA C-MAPSS feature schema")

    return {
        "ai4i_missing_features": ai4i_missing,
        "nasa_missing_features": nasa_missing,
        "compatible_models": compatible_models,
    }


def discover_target_columns(df):
    targets = []

    for column in df.columns:
        name = str(column).strip().lower()

        normalised = re.sub(r"\s+", " ", name)

        is_known_target = normalised in KNOWN_TARGETS

        is_target_pattern = (
            name.endswith("_target")
            or name.endswith("_label")
            or name.startswith("target_")
            or name.startswith("label_")
        )

        if is_known_target or is_target_pattern:
            targets.append(column)

    return targets


def get_excluded_columns(df, selected_target):
    excluded = set()

    for column in df.columns:
        name = str(column).strip().lower()

        if any(token in name for token in EXCLUDED_FEATURE_TOKENS):
            excluded.add(column)

        if (
            name.endswith("_target")
            or name.endswith("_label")
            or name.startswith("target_")
            or name.startswith("label_")
        ):
            excluded.add(column)

    excluded.add(selected_target)

    return excluded


def prepare_features(df, target_column):
    excluded = get_excluded_columns(df, target_column)

    candidate_columns = [
        column for column in df.columns
        if column not in excluded
    ]

    X = df[candidate_columns].copy()

    numeric_columns = X.select_dtypes(include="number").columns

    if len(numeric_columns):
        X[numeric_columns] = X[numeric_columns].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    X = X.dropna(axis=1, how="all")

    constant_columns = [
        column for column in X.columns
        if X[column].nunique(dropna=True) <= 1
    ]

    X = X.drop(columns=constant_columns)

    return X


def build_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        column for column in X.columns
        if column not in numeric_features
    ]

    transformers = []

    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric_features,
        ))

    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            categorical_features,
        ))

    if not transformers:
        raise ValueError("No usable input features found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_models():
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    except Exception as exc:
        print(f"XGBoost unavailable: {exc}")

    return models


def evaluate_model(
    model_name,
    estimator,
    X_train,
    X_test,
    y_train,
    y_test,
    label_encoder,
    test_indices,
):
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", estimator),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = None

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(label_encoder.classes_)),
    )

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision_weighted": float(
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": [
            str(label) for label in label_encoder.classes_
        ],
    }

    if (
        probabilities is not None
        and len(label_encoder.classes_) == 2
    ):
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_test, probabilities[:, 1])
            )
        except Exception:
            metrics["roc_auc"] = None

    actual_labels = label_encoder.inverse_transform(
        np.asarray(y_test, dtype=int)
    )

    predicted_labels = label_encoder.inverse_transform(
        np.asarray(predictions, dtype=int)
    )

    prediction_data = pd.DataFrame({
        "source_row_index": test_indices,
        "actual": actual_labels,
        "predicted": predicted_labels,
        "correct": (
            np.asarray(actual_labels)
            == np.asarray(predicted_labels)
        ),
    })

    if probabilities is not None:
        for class_index, class_name in enumerate(
            label_encoder.classes_
        ):
            prediction_data[
                f"probability_{class_name}"
            ] = probabilities[:, class_index]

    return metrics, prediction_data


def evaluate_target(df, dataset_path, target_column):
    print("\n" + "-" * 65)
    print(f"TARGET: {target_column}")
    print("-" * 65)

    result = {
        "target": str(target_column),
        "status": "ERROR",
    }

    y_original = df[target_column]

    valid_target = y_original.notna()

    X = prepare_features(df, target_column)
    X = X.loc[valid_target]

    y_original = y_original.loc[valid_target]

    if pd.api.types.is_numeric_dtype(y_original):
        numeric_target = pd.to_numeric(
            y_original,
            errors="coerce",
        )

        finite_mask = np.isfinite(numeric_target)

        X = X.loc[finite_mask]
        y_original = y_original.loc[finite_mask]

    if X.empty:
        result["error"] = "No usable feature rows."
        return result

    if X.shape[1] == 0:
        result["error"] = "No usable feature columns."
        return result

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_original.astype(str)
    )

    class_counts = pd.Series(
        y_original.astype(str)
    ).value_counts()

    result["usable_rows"] = int(len(X))
    result["feature_count"] = int(X.shape[1])
    result["class_distribution"] = {
        str(label): int(count)
        for label, count in class_counts.items()
    }
    result["classes"] = [
        str(label) for label in label_encoder.classes_
    ]

    print(f"Usable labelled rows : {len(X)}")
    print(f"Input features       : {X.shape[1]}")
    print(f"Number of classes    : {len(label_encoder.classes_)}")

    if len(class_counts) < 2:
        result["error"] = "Only one target class exists."
        return result

    if class_counts.min() < 2:
        result["error"] = (
            "At least one class has fewer than two records."
        )
        return result

    row_indices = X.index.to_numpy()

    try:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            train_indices,
            test_indices,
        ) = train_test_split(
            X,
            y,
            row_indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    except ValueError as exc:
        result["error"] = str(exc)
        return result

    print(f"Training records    : {len(X_train)}")
    print(f"Testing records     : {len(X_test)}")

    model_results = []
    predictions_by_model = {}

    for model_name, estimator in build_models().items():
        print(f"\nEvaluating {model_name}...")

        try:
            metrics, prediction_data = evaluate_model(
                model_name=model_name,
                estimator=estimator,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                label_encoder=label_encoder,
                test_indices=test_indices,
            )

            model_results.append(metrics)
            predictions_by_model[model_name] = prediction_data

            print(f"Accuracy          : {metrics['accuracy']:.4f}")
            print(
                f"Balanced accuracy : "
                f"{metrics['balanced_accuracy']:.4f}"
            )
            print(
                f"Precision         : "
                f"{metrics['precision_weighted']:.4f}"
            )
            print(
                f"Recall            : "
                f"{metrics['recall_weighted']:.4f}"
            )
            print(
                f"F1 score          : "
                f"{metrics['f1_weighted']:.4f}"
            )
            print("Confusion matrix:")
            print(np.asarray(metrics["confusion_matrix"]))

            if metrics.get("roc_auc") is not None:
                print(f"ROC-AUC           : {metrics['roc_auc']:.4f}")

        except Exception as exc:
            print(f"MODEL ERROR: {model_name}: {exc}")

    if not model_results:
        result["error"] = "All model evaluations failed."
        return result

    model_results.sort(
        key=lambda item: item["f1_weighted"],
        reverse=True,
    )

    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        dataset_path.stem,
    )

    safe_target = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(target_column),
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, prediction_data in predictions_by_model.items():
        safe_model = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            model_name,
        )

        prediction_file = (
            REPORT_DIR
            / f"{safe_name}_{safe_target}_{safe_model}_predictions.csv"
        )

        prediction_data.to_csv(prediction_file, index=False)

    best_model = model_results[0]

    result.update({
        "status": "EVALUATED",
        "split": {
            "method": "Stratified train_test_split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "training_rows": int(len(X_train)),
            "testing_rows": int(len(X_test)),
        },
        "models": model_results,
        "best_model_by_weighted_f1": best_model["model"],
        "evaluation_warning": (
            "Metrics measure agreement with supplied labels. "
            "The labels have not been independently verified "
            "as real-world ground truth."
        ),
    })

    print("\nBEST MODEL BY WEIGHTED F1:")
    print(f"Model    : {best_model['model']}")
    print(f"Accuracy : {best_model['accuracy']:.4f}")
    print(f"F1 score : {best_model['f1_weighted']:.4f}")

    return result


def verify_dataset(path):
    print("\n" + "=" * 70)
    print(f"DATASET: {path.name}")
    print("=" * 70)

    result = {
        "file": path.name,
        "status": "ERROR",
    }

    try:
        df = load_dataset(path)

        if df.empty:
            raise ValueError("Dataset is empty.")

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        if df.columns.duplicated().any():
            raise ValueError("Duplicate column names found.")

        missing_values = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        numeric_count, non_finite = check_numeric_quality(df)

        schema = inspect_schema(df)
        targets = discover_target_columns(df)

        result.update({
            "rows": int(len(df)),
            "columns": [str(column) for column in df.columns],
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "numeric_columns": numeric_count,
            "non_finite_values": non_finite,
            "compatible_models": schema["compatible_models"],
            "ai4i_missing_features": schema["ai4i_missing_features"],
            "nasa_missing_features": schema["nasa_missing_features"],
            "schema_match": bool(schema["compatible_models"]),
            "recognised_targets": [str(t) for t in targets],
            "status": "INSPECTED",
        })

        print(f"Rows                 : {len(df)}")
        print(f"Columns              : {len(df.columns)}")
        print(f"Missing values       : {missing_values}")
        print(f"Duplicate rows       : {duplicate_rows}")
        print(f"Numeric columns      : {numeric_count}")
        print(f"Non-finite values    : {non_finite}")

        print("\nEXISTING MODEL COMPATIBILITY:")

        if schema["compatible_models"]:
            for model in schema["compatible_models"]:
                print(f"  [MATCH] {model}")
        else:
            print("  No recognised model schema matched.")

        print("\nRECOGNISED TARGET COLUMNS:")

        if targets:
            for target in targets:
                print(f"  [FOUND] {target}")
        else:
            print("  No recognised supervised target found.")

        target_results = []

        for target in targets:
            target_results.append(
                evaluate_target(df, path, target)
            )

        result["target_evaluations"] = target_results

        successful = [
            item for item in target_results
            if item["status"] == "EVALUATED"
        ]

        print(
            f"\nCompleted target evaluations: "
            f"{len(successful)}/{len(target_results)}"
        )

    except Exception as exc:
        result["error"] = str(exc)
        print(f"\nDATASET ERROR: {exc}")

    return result


def main():
    print("\nEXTERNAL MULTI-DATASET VERIFICATION")
    print("=" * 70)

    print("Existing AI4I and NASA model artifacts will not be modified.")
    print("External datasets will be evaluated independently.")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    supported_extensions = {
        ".csv",
        ".txt",
        ".xlsx",
        ".xls",
    }

    files = sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )

    if not files:
        print(f"\nNo datasets found in:\n{DATASET_DIR}")
        return

    print(f"\nDatasets found: {len(files)}")

    results = [
        verify_dataset(path)
        for path in files
    ]

    report = {
        "dataset_folder": str(DATASET_DIR),
        "total_datasets": len(results),
        "existing_models_modified": False,
        "results": results,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            default=str,
        )

    print("\n" + "=" * 70)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 70)

    for result in results:
        print(f"\nDataset: {result['file']}")
        print(f"Status: {result['status']}")

        for evaluation in result.get("target_evaluations", []):
            print(
                f"  {evaluation['target']}: "
                f"{evaluation['status']}"
            )

            if evaluation.get("best_model_by_weighted_f1"):
                print(
                    "  Best model: "
                    f"{evaluation['best_model_by_weighted_f1']}"
                )

    print(f"\nDetailed report saved to:\n{REPORT_FILE}")
    print("\nExisting AI4I/NASA model artifacts were not changed.")
    print("EXTERNAL DATASET EVALUATION COMPLETED")


if __name__ == "__main__":
    main()

