
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.base import clone


def build_anomaly_models(preprocessor):
    """
    Build unsupervised anomaly detection models.

    All models train without failure labels.
    The same preprocessing pipeline is used for each model.
    """

    # 1. Isolation Forest
    isolation_forest = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("detector", IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",
            max_features=1.0,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 2. Local Outlier Factor
    lof = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("detector", LocalOutlierFactor(
            n_neighbors=20,
            algorithm="auto",
            contamination="auto",
            novelty=True,
            n_jobs=-1
        ))
    ])

    # 3. One-Class SVM
    one_class_svm = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("detector", OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.05,
            cache_size=1000,
            tol=1e-3,
            max_iter=10000
        ))
    ])

    return {
        "Isolation Forest": isolation_forest,
        "Local Outlier Factor": lof,
        "One-Class SVM": one_class_svm
    }


def get_anomaly_predictions(model, X):
    """
    Convert sklearn detector outputs into:

    0 = Normal
    1 = Anomaly

    sklearn uses:
     1 = Normal
    -1 = Anomaly
    """

    predictions = model.predict(X)

    return (predictions == -1).astype(int)


def get_anomaly_scores(model, X):
    """
    Return anomaly scores.

    Higher score means more anomalous behaviour.
    """

    return -model.decision_function(X)
