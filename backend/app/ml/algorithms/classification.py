
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from xgboost import XGBClassifier


def build_classification_models(preprocessor, y_train):

    normal_count = int((y_train == 0).sum())
    failure_count = int((y_train == 1).sum())

    if failure_count == 0:
        raise ValueError("No failure samples found.")

    scale_pos_weight = normal_count / failure_count

    print("\nCLASS DISTRIBUTION")
    print("Normal:", normal_count)
    print("Failure:", failure_count)
    print("Scale positive weight:", round(scale_pos_weight, 2))

    # Random Forest
    rf = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # XGBoost
    xgb = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            reg_lambda=1.0,
            random_state=42,
            n_jobs=2
        ))
    ])

    return {
        "Random Forest": rf,
        "XGBoost": xgb
    }
