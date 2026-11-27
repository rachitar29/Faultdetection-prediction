
from app.ml.algorithms.model_registry import (
    list_registered_models,
    REGISTRY_DIR
)


def test_registry():

    print("\nMODEL REGISTRY TEST")
    print("=" * 50)

    print("Registry directory:")
    print(REGISTRY_DIR)

    models = list_registered_models()

    if not models:
        print("\nNo models registered yet.")
        print("This is expected before registration.")
        return

    print("\nREGISTERED MODELS")

    for metadata in models:
        print("-" * 40)
        print("Name:", metadata.get("model_name"))
        print("Type:", metadata.get("model_type"))
        print("Threshold:", metadata.get("threshold"))
        print("Features:", metadata.get("feature_names"))

    print("\nREGISTRY TEST COMPLETED")


if __name__ == "__main__":
    test_registry()
