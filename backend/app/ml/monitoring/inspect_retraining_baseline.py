
"""Inspect registered AI4I XGBoost models before controlled retraining.

Read-only operation:
- Lists registered model metadata.
- Finds AI4I XGBoost candidates.
- Checks whether each model can be loaded.
- Displays available evaluation metrics.
- Does not modify the registry.
"""

import json

from app.ml.algorithms.model_registry import (
    list_registered_models,
    load_model_bundle,
)


def is_xgboost_model(metadata):
    """Identify XGBoost model entries."""

    name = str(metadata.get("model_name", "")).lower()
    model_type = str(metadata.get("model_type", "")).lower()

    return "xgboost" in name and (
        "classification" in model_type
        or "xgboost" in name
    )


def is_ai4i_model(metadata):
    """Identify models associated with the AI4I dataset."""

    name = str(metadata.get("model_name", "")).lower()
    description = str(metadata.get("description", "")).lower()

    return "ai4i" in name or "ai4i" in description


def main():
    print("\nMODEL REGISTRY INSPECTION")
    print("=" * 65)

    registered_models = list_registered_models()

    if not registered_models:
        print("No registered models were found.")
        return

    print("Total registered model entries:", len(registered_models))

    # Find AI4I-specific XGBoost candidates.
    candidates = [
        metadata
        for metadata in registered_models
        if is_xgboost_model(metadata)
        and is_ai4i_model(metadata)
    ]

    # Also identify the previously tested generic XGBoost model.
    generic_candidates = [
        metadata
        for metadata in registered_models
        if str(metadata.get("model_name", "")).lower() == "xgboost"
    ]

    print("\nAI4I XGBOOST CANDIDATES")
    print("-" * 65)

    if not candidates:
        print("No AI4I-specific XGBoost model was found.")
    else:
        for index, metadata in enumerate(candidates, start=1):
            model_name = metadata.get("model_name", "Unknown")

            print(f"\nCandidate {index}: {model_name}")
            print("Model type:", metadata.get("model_type"))
            print("Threshold:", metadata.get("threshold"))
            print("Created:", metadata.get("created_at_utc"))
            print("Features:", metadata.get("feature_names"))

            print("Metrics:")
            print(
                json.dumps(
                    metadata.get("metrics", {}),
                    indent=4,
                )
            )

            # Verify that the registered bundle can be loaded.
            try:
                bundle = load_model_bundle(model_name)

                if "model" not in bundle or "metadata" not in bundle:
                    raise ValueError(
                        "Loaded bundle is missing model or metadata."
                    )

                print("Load verification: PASSED")

            except Exception as error:
                print("Load verification: FAILED")
                print("Reason:", error)

    print("\nGENERIC XGBOOST MODEL")
    print("-" * 65)

    if not generic_candidates:
        print("No generic XGBoost entry was found.")
    else:
        for metadata in generic_candidates:
            model_name = metadata.get("model_name", "Unknown")

            print("\nModel:", model_name)
            print("Threshold:", metadata.get("threshold"))
            print("Metrics:")
            print(
                json.dumps(
                    metadata.get("metrics", {}),
                    indent=4,
                )
            )

            try:
                load_model_bundle(model_name)
                print("Load verification: PASSED")
            except Exception as error:
                print("Load verification: FAILED")
                print("Reason:", error)

    print("\nINSPECTION SUMMARY")
    print("=" * 65)
    print("AI4I XGBoost candidates:", len(candidates))
    print("Generic XGBoost candidates:", len(generic_candidates))
    print("Registry modified: NO")
    print("\nREGISTRY INSPECTION COMPLETED")


if __name__ == "__main__":
    main()
