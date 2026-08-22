"""
Full Pipeline: Model A -> Model B
====================================
End-to-end inference: raw fluorescence features -> Model A (contamination
level) -> Model B (Substance X dose recommendation).

This mirrors what the FRDM-MCXN236 firmware will eventually do on-device:
two small models chained back-to-back, running on real (or here, manually
entered / synthetic) photodiode-derived data.

Usage:
    # Interactive mode
    python3 src/full_pipeline.py

    # CLI mode (single reading)
    python3 src/full_pipeline.py --peak 0.85 --mean 0.18 --rise 0.002 --decay 0.006 --energy 0.009

    # Batch mode (CSV with the 5 Model A feature columns)
    python3 src/full_pipeline.py --csv data/manual_test_cases.csv
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Model A (contamination detection)
MODEL_A_PATH = os.path.join(MODELS_DIR, "nn_baseline.keras")
SCALER_A_PATH = os.path.join(MODELS_DIR, "feature_scaler.joblib")

# Model B (dose prediction)
MODEL_B_PATH = os.path.join(MODELS_DIR, "dosing_model.keras")
SCALER_B_PATH = os.path.join(MODELS_DIR, "dosing_scaler.joblib")

FEATURE_NAMES_A = ["peak_intensity", "mean_intensity", "rise_time", "decay_time", "energy"]

# Calibrated range Model A was trained on (data/generate_dataset.py CONC_RANGE_MGL)
CONTAMINATION_RANGE_MGL = (0.0, 100.0)


def load_models():
    model_a = tf.keras.models.load_model(MODEL_A_PATH)
    scaler_a = joblib.load(SCALER_A_PATH)
    model_b = tf.keras.models.load_model(MODEL_B_PATH)
    scaler_b = joblib.load(SCALER_B_PATH)
    return model_a, scaler_a, model_b, scaler_b


def run_pipeline(model_a, scaler_a, model_b, scaler_b, raw_features):
    """raw_features: list of 5 floats in FEATURE_NAMES_A order."""
    # --- Model A: features -> contamination ---
    x_a = np.array([raw_features], dtype=np.float32)
    x_a_scaled = scaler_a.transform(x_a).astype(np.float32)
    contamination_raw = float(model_a.predict(x_a_scaled, verbose=0).flatten()[0])
    contamination_raw = max(contamination_raw, 0.0)  # concentration can't be negative

    # --- Flag + clip out-of-calibrated-range predictions ---
    lo, hi = CONTAMINATION_RANGE_MGL
    out_of_range = contamination_raw > hi or contamination_raw < lo
    contamination = min(max(contamination_raw, lo), hi)

    # --- Model B: contamination -> dose ---
    x_b = np.array([[contamination]], dtype=np.float32)
    x_b_scaled = scaler_b.transform(x_b).astype(np.float32)
    dose = float(model_b.predict(x_b_scaled, verbose=0).flatten()[0])
    dose = max(dose, 0.0)  # dose can't be negative

    return contamination_raw, contamination, dose, out_of_range


def print_result(raw_features, contamination_raw, contamination, dose, out_of_range, label=""):
    print(f"\n{label}" if label else "")
    print(f"  Input features        : {dict(zip(FEATURE_NAMES_A, raw_features))}")
    if out_of_range:
        print(f"  [!] WARNING: raw prediction {contamination_raw:.2f} mg/L is outside the "
              f"calibrated range {CONTAMINATION_RANGE_MGL} - clipped, treat as approximate.")
    print(f"  -> Predicted contamination : {contamination:.2f} mg/L")
    print(f"  -> Recommended Substance X dose : {dose:.2f} mg/L")


def interactive_mode(model_a, scaler_a, model_b, scaler_b):
    print("Enter photodiode-derived feature values manually.\n")
    raw_features = []
    for name in FEATURE_NAMES_A:
        val = float(input(f"  {name}: "))
        raw_features.append(val)

    contamination_raw, contamination, dose, out_of_range = run_pipeline(
        model_a, scaler_a, model_b, scaler_b, raw_features
    )
    print_result(raw_features, contamination_raw, contamination, dose, out_of_range, label="Pipeline result:")


def csv_mode(model_a, scaler_a, model_b, scaler_b, csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_NAMES_A if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    print(f"Running {len(df)} test cases from {csv_path}...")
    results = []
    for i, row in df.iterrows():
        raw_features = [row[c] for c in FEATURE_NAMES_A]
        contamination_raw, contamination, dose, out_of_range = run_pipeline(
            model_a, scaler_a, model_b, scaler_b, raw_features
        )
        print_result(raw_features, contamination_raw, contamination, dose, out_of_range, label=f"Row {i}:")
        results.append(
            {
                "row": i,
                "predicted_contamination_raw_mgl": contamination_raw,
                "predicted_contamination_mgl": contamination,
                "recommended_dose_mgl": dose,
                "out_of_calibrated_range": out_of_range,
            }
        )

    out_path = os.path.join(os.path.dirname(csv_path), "pipeline_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved batch results -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: Model A -> Model B")
    parser.add_argument("--peak", type=float, help="peak_intensity")
    parser.add_argument("--mean", type=float, help="mean_intensity")
    parser.add_argument("--rise", type=float, help="rise_time")
    parser.add_argument("--decay", type=float, help="decay_time")
    parser.add_argument("--energy", type=float, help="energy")
    parser.add_argument("--csv", type=str, help="Path to CSV of test cases")
    args = parser.parse_args()

    model_a, scaler_a, model_b, scaler_b = load_models()

    if args.csv:
        csv_mode(model_a, scaler_a, model_b, scaler_b, args.csv)
    elif all(v is not None for v in [args.peak, args.mean, args.rise, args.decay, args.energy]):
        raw_features = [args.peak, args.mean, args.rise, args.decay, args.energy]
        contamination_raw, contamination, dose, out_of_range = run_pipeline(
            model_a, scaler_a, model_b, scaler_b, raw_features
        )
        print_result(raw_features, contamination_raw, contamination, dose, out_of_range, label="CLI input result:")
    else:
        interactive_mode(model_a, scaler_a, model_b, scaler_b)


if __name__ == "__main__":
    main()