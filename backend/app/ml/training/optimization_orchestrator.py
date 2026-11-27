
from app.ml.training.model_optimization import optimize_multiple_datasets


DATASET_SPECS = [
    {
        "name": "AI4I",
        "dataset_path": "ml_data/raw/ai4i2020.csv",
        "config_path": (
            "app/ml/configurations/dataset_configs/ai4i_config.json"
        )
    },
    {
        "name": "Alternative Industrial Sensors",
        "dataset_path": (
            "ml_data/validity_test/alternative_sensors/"
            "alternative_sensor_clean.csv"
        ),
        "config_path": (
            "app/ml/configurations/dataset_configs/"
            "alternative_sensor_config.json"
        )
    }
]


if __name__ == "__main__":
    optimize_multiple_datasets(DATASET_SPECS)
