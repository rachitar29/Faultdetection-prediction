
from app.ml.algorithms.classification import (
    build_classification_models
)

from app.ml.algorithms.anomaly_detection import (
    build_anomaly_models
)


def select_classification_models(preprocessor, y_train):
    """Build supervised classification models."""

    if y_train is None:
        raise ValueError(
            "Classification requires target labels."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "Classification requires at least two target classes."
        )

    return build_classification_models(
        preprocessor=preprocessor,
        y_train=y_train
    )


def select_anomaly_models(preprocessor):
    """Build unsupervised anomaly detection models."""

    return build_anomaly_models(
        preprocessor=preprocessor
    )


def select_models(task, preprocessor, y_train=None):
    """
    Select models based on the requested ML task.

    Supported tasks:
    - classification
    - anomaly_detection
    - all
    """

    task = task.lower().strip()

    if task == "classification":
        return select_classification_models(
            preprocessor,
            y_train
        )

    if task == "anomaly_detection":
        return select_anomaly_models(
            preprocessor
        )

    if task == "all":

        models = {}

        if y_train is not None and y_train.nunique() >= 2:
            models.update(
                select_classification_models(
                    preprocessor,
                    y_train
                )
            )

        models.update(
            select_anomaly_models(preprocessor)
        )

        return models

    raise ValueError(
        f"Unsupported ML task: {task}"
    )
