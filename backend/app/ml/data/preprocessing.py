
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Sensor features
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Categorical features
CATEGORICAL_FEATURES = [
    "Type"
]

# Prediction target
TARGET = "Machine failure"

# Columns that must not enter binary classification
EXCLUDED_COLUMNS = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]


def separate_features_target(df):
    """
    Separate input sensor features and target label.
    """

    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    print("\nFEATURE AND TARGET SEPARATION")
    print("=" * 50)

    print("Input features:", list(X.columns))
    print("Target:", TARGET)

    print("\nExcluded columns:", EXCLUDED_COLUMNS)

    return X, y


def build_preprocessor():
    """
    Build preprocessing pipelines.

    Numerical:
        Missing value imputation + StandardScaler

    Categorical:
        Most frequent imputation + OneHotEncoder
    """

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


def split_dataset(X, y, test_size=0.20, random_state=42):
    """
    Split dataset into training and testing sets.

    Stratification preserves failure distribution.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("\nDATASET SPLITTING")
    print("=" * 50)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


def prepare_data(df):
    """
    Complete dataset preparation.

    Returns raw training/testing data and an
    unfitted preprocessing pipeline.

    The preprocessing pipeline will be fitted
    inside each model's training pipeline.
    """

    X, y = separate_features_target(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    preprocessor = build_preprocessor()

    print("\nPREPROCESSING PIPELINE CREATED")
    print("Numerical features:", len(NUMERIC_FEATURES))
    print("Categorical features:", len(CATEGORICAL_FEATURES))
    print("Target:", TARGET)

    print("\nData preparation completed.")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":

    from app.ml.data.data_loader import load_dataset
    from app.ml.data.validation import validate_dataset

    df = load_dataset()

    validate_dataset(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data(df)
