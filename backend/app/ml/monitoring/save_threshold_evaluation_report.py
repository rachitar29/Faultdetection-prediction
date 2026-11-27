             
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    backend_dir = Path(__file__).resolve().parents[3]

    output_dir = backend_dir / "ml_artifacts" / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "evaluation_name": "XGBoost Balanced Threshold Optimization",
        "model_name": "XGBoost",
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),

        "dataset": {
            "name": "AI4I 2020 Predictive Maintenance Dataset",
            "reference_samples": 8000,
            "independent_test_samples": 2000,
            "test_size": 0.20,
            "random_state": 42,
            "cross_validation_folds": 5
        },

        "optimization": {
            "objective": "Maximum F1-score",
            "minimum_precision": 0.80,
            "minimum_recall": None,
            "cross_validation_constraints_satisfied": True
        },

        "thresholds": {
            "original_threshold": 0.884595513343811,
            "optimized_candidate_threshold": 0.824862539768219,
            "production_threshold_modified": False
        },

        "cross_validation_metrics": {
            "accuracy": 0.982625,
            "precision": 0.80,
            "recall": 0.6494464944649446,
            "f1": 0.7169042769857433,
            "roc_auc": 0.9658123738696307
        },

        "independent_test": {
            "original_threshold": {
                "accuracy": 0.9855,
                "precision": 0.8823529411764706,
                "recall": 0.6617647058823529,
                "f1": 0.7563025210084033,
                "roc_auc": 0.9742342589209597,
                "false_negatives": 23,
                "false_positives": 6
            },

            "optimized_threshold": {
                "accuracy": 0.9865,
                "precision": 0.8596491228070176,
                "recall": 0.7205882352941176,
                "f1": 0.784,
                "roc_auc": 0.9742342589209597,
                "false_negatives": 19,
                "false_positives": 8
            }
        },

        "constraint_verification": {
            "cross_validation_precision_at_least_0_80": True,
            "independent_test_precision_at_least_0_80": True,
            "independent_test_precision": 0.8596491228070176
        },

        "decision": {
            "status": "VALIDATED_CANDIDATE",
            "recommended_candidate_threshold": 0.824862539768219,
            "production_promotion_approved": False,
            "notes": (
                "The optimized threshold improved independent-test recall "
                "and F1-score while maintaining precision above 0.80. "
                "Production model and registered threshold remain unchanged."
            )
        }
    }

    report_path = output_dir / "xgboost_threshold_evaluation.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("=" * 65)
    print("XGBOOST THRESHOLD EVALUATION REPORT SAVED")
    print("=" * 65)
    print(f"Report: {report_path}")
    print(f"Candidate threshold: {report['thresholds']['optimized_candidate_threshold']}")
    print(f"Independent test precision: {report['independent_test']['optimized_threshold']['precision']:.4f}")
    print(f"Independent test recall: {report['independent_test']['optimized_threshold']['recall']:.4f}")
    print(f"Independent test F1: {report['independent_test']['optimized_threshold']['f1']:.4f}")
    print("Production model modified: NO")
    print("Production threshold modified: NO")


if __name__ == "__main__":
    main()
