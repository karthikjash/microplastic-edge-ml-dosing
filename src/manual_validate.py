"""
Manual Synthetic-Value Validation Harness
============================================
Simulates what the FRDM-MCXN236 will do on-device: takes 5 feature
values (peak_intensity, mean_intensity, rise_time, decay_time, energy),
scales them, quantizes to int8, runs through the deployed TFLite int8
model, and prints the predicted concentration (mg/L).

Also runs the same input through the original float32 Keras model so
you can compare and check quantization isn't drifting on realistic
single inputs.

Usage:
    # Interactive mode (prompts you for each feature)
    python3 src/manual_validate.py

    # CLI mode (single reading)
    python3 src/manual_validate.py --peak 0.85 --mean 0.18 --rise 0.002 --decay 0.006 --energy 0.009

    # Batch mode (CSV of test cases, same 5 columns as training data)
    python3 src/manual_validate.py --csv data/manual_test_cases.csv
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "nn_baseline.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.joblib")
TFLITE_INT8_PATH = os.path.join(MODELS_DIR, "nn_baseline_int8.tflite")

FEATURE_NAMES = ["peak_intensity", "mean_intensity", "rise_time", "decay_time", "energy"]


def load_models():
    keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    interpreter = tf.lite.Interpreter(model_path=TFLITE_INT8_PATH)
    interpreter.allocate_tensors()
    return keras_model, scaler, interpreter


def predict_keras(keras_model, scaler, raw_features):
    x = np.array([raw_features], dtype=np.float32)
    x_scaled = scaler.transform(x).astype(np.float32)
    pred = keras_model.predict(x_scaled, verbose=0).flatten()[0]
    return float(pred)


def predict_tflite_int8(interpreter, scaler, raw_features):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    x = np.array([raw_features], dtype=np.float32)
    x_scaled = scaler.transform(x).astype(np.float32)

    in_scale, in_zero_point = input_details["quantization"]
    x_q = (x_scaled / in_scale + in_zero_point).astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], x_q)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details["index"])

    out_scale, out_zero_point = output_details["quantization"]
    pred = (out.astype(np.float32) - out_zero_point) * out_scale
    return float(pred.flatten()[0])


def run_single(keras_model, scaler, interpreter, raw_features, label=""):
    keras_pred = predict_keras(keras_model, scaler, raw_features)
    tflite_pred = predict_tflite_int8(interpreter, scaler, raw_features)
    diff = abs(keras_pred - tflite_pred)

    print(f"\n{label}" if label else "")
    print(f"  Input features: {dict(zip(FEATURE_NAMES, raw_features))}")
    print(f"  Keras (float32) prediction : {keras_pred:.3f} mg/L")
    print(f"  TFLite (int8)   prediction : {tflite_pred:.3f} mg/L")
    print(f"  Difference                 : {diff:.3f} mg/L")
    return keras_pred, tflite_pred


def interactive_mode(keras_model, scaler, interpreter):
    print("Enter photodiode-derived feature values manually.")
    print("(Reference ranges from synthetic dataset: peak ~0-1.5, mean ~0-0.45,")
    print(" rise_time ~0-0.005s, decay_time ~0-0.02s, energy ~0-0.022)\n")

    raw_features = []
    for name in FEATURE_NAMES:
        val = float(input(f"  {name}: "))
        raw_features.append(val)

    run_single(keras_model, scaler, interpreter, raw_features, label="Manual entry result:")


def csv_mode(keras_model, scaler, interpreter, csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    print(f"Running {len(df)} test cases from {csv_path}...")
    results = []
    for i, row in df.iterrows():
        raw_features = [row[c] for c in FEATURE_NAMES]
        keras_pred, tflite_pred = run_single(
            keras_model, scaler, interpreter, raw_features, label=f"Row {i}:"
        )
        results.append({"row": i, "keras_pred": keras_pred, "tflite_int8_pred": tflite_pred})

    out_path = os.path.join(os.path.dirname(csv_path), "manual_validation_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved batch results -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Manual validation harness for deployed model")
    parser.add_argument("--peak", type=float, help="peak_intensity")
    parser.add_argument("--mean", type=float, help="mean_intensity")
    parser.add_argument("--rise", type=float, help="rise_time")
    parser.add_argument("--decay", type=float, help="decay_time")
    parser.add_argument("--energy", type=float, help="energy")
    parser.add_argument("--csv", type=str, help="Path to CSV of test cases")
    args = parser.parse_args()

    keras_model, scaler, interpreter = load_models()

    if args.csv:
        csv_mode(keras_model, scaler, interpreter, args.csv)
    elif all(v is not None for v in [args.peak, args.mean, args.rise, args.decay, args.energy]):
        raw_features = [args.peak, args.mean, args.rise, args.decay, args.energy]
        run_single(keras_model, scaler, interpreter, raw_features, label="CLI input result:")
    else:
        interactive_mode(keras_model, scaler, interpreter)


if __name__ == "__main__":
    main()