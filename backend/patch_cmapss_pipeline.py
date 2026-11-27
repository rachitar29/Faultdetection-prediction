from pathlib import Path

path = Path("app/ml/maintenance/cmapss_rul_pipeline.py")
source = path.read_text(encoding="utf-8")

old = '''    test_metrics, predictions_df = evaluate_test_engines(
        best_model,
        test_df,
        actual_rul
    )

    best_validation_metrics = validation_results[best_name]

    register_best_model(
        best_model,
        best_name,
        best_validation_metrics,
        test_metrics
    )
'''

new = '''    # Retrain the selected algorithm on all available training engines.
    # The validation split is used only for model selection.
    print("\\\\nRefitting selected model using all training engines...")

    final_model = build_models()[best_name]

    X_full_train = prepare_features(train_df)
    y_full_train = train_df["RUL"].to_numpy()

    final_model.fit(X_full_train, y_full_train)

    print(f"Final training rows: {len(X_full_train)}")
    print(f"Final training engines: {train_df['unit'].nunique()}")

    # Evaluate the refitted model on the untouched NASA test engines.
    test_metrics, predictions_df = evaluate_test_engines(
        final_model,
        test_df,
        actual_rul
    )

    best_validation_metrics = validation_results[best_name]

    register_best_model(
        final_model,
        best_name,
        best_validation_metrics,
        test_metrics
    )
'''

if source.count(old) != 1:
    raise SystemExit(
        "PATCH STOPPED: Expected code block was not found exactly once. "
        "Original file was not changed."
    )

path.write_text(source.replace(old, new), encoding="utf-8")
print("Pipeline updated successfully.")
