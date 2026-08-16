"""
TFLite Conversion + INT8 Quantization
=======================================
Converts the trained Keras NN baseline (models/nn_baseline.keras) into
TFLite format, in two flavors:

    1. Float32 TFLite  - sanity-check conversion, no quantization
    2. INT8 TFLite     - fully quantized (weights + activations), the
                          format needed for the FRDM-MCXN236 eIQ Neutron NPU

Then compares accuracy (MAE/RMSE/R2) and file size across:
    - Original Keras model (float32)
    - TFLite float32
    - TFLite int8

Outputs:
    models/nn_baseline_float32.tflite
    models/nn_baseline_int8.tflite
    results/quantization_comparison.json

Usage:
    python3 src/convert_tflite.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "synthetic_fluorescence_dataset.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
RANDOM_STATE = 42

FEATURE_COLS = ["peak_intensity", "mean_intensity", "rise_time", "decay_time", "energy"]
TARGET_COL = "concentration_mgl"

KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "nn_baseline.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.joblib")
TFLITE_FLOAT32_PATH = os.path.join(MODELS_DIR, "nn_baseline_float32.tflite")
TFLITE_INT8_PATH = os.path.join(MODELS_DIR, "nn_baseline_int8.tflite")

def load_test_split():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred):
    return {
        "mae" : float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2"  : float(r2_score(y_true, y_pred)),
    }

def convert_float32(keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    with open(TFLITE_FLOAT32_PATH, "wb") as f:
        f.write(tflite_model)
    return TFLITE_FLOAT32_PATH


"""Full int8 quantization: weights AND activations, with int8 input/output
    so the model is ready for the eIQ Neutron NPU (which expects int8 tensors,
    not float32 in/out with quantized-only weights)."""

def convert_int8(keras_model, X_train_scaled):
    def representative_dataset():
        #using a subset of the scaled train data to calibrate quantization ranges

        n_calib = min(300, len(X_train_scaled))
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train_scaled), size=n_calib, replace=False)

        for i in idx:
            sample = X_train_scaled[i:i+1].astype(np.float32)
            yield [sample]


    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(TFLITE_INT8_PATH, "wb") as f:
        f.write(tflite_model)
    return TFLITE_INT8_PATH


def run_tflite_inference(tflite_path, X, is_int8=False):
    """ 
    run a TFLite model (float32 or int8) over x and return predictions
    in the original unscaled prediction space
    """

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    preds = []
    for i in range(len(X)):
        sample = X[i : i + 1].astype(np.float32)

        if is_int8:
            in_scale, in_zero_point = input_details["quantization"]
            sample_q = (sample / in_scale + in_zero_point).astype(input_details["dtype"])
            interpreter.set_tensor(input_details["index"], sample_q)
        else:
            interpreter.set_tensor(input_details["index"], sample)

        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])

        if is_int8:
            out_scale, out_zero_point = output_details["quantization"]
            out = (out.astype(np.float32) - out_zero_point) * out_scale

        preds.append(float(out.flatten()[0]))

    return np.array(preds)


def main():
    print("Loading Keras model + scaler...")
    if not os.path.exists(KERAS_MODEL_PATH):
        raise FileNotFoundError(
            f"{KERAS_MODEL_PATH} not found. Run src/train_baseline.py first."
        )
    keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X_train, X_test, y_train, y_test = load_test_split()
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    results = {}

    # --- 1. Original Keras (float32) ---
    print("\nEvaluating original Keras model...")
    keras_preds = keras_model.predict(X_test_scaled, verbose=0).flatten()
    results["keras_float32"] = evaluate(y_test, keras_preds)
    results["keras_float32"]["size_kb"] = round(
        os.path.getsize(KERAS_MODEL_PATH) / 1024, 2
    )
    print(f"  MAE={results['keras_float32']['mae']:.3f}  R2={results['keras_float32']['r2']:.3f}"
          f"  size={results['keras_float32']['size_kb']} KB")

    # --- 2. TFLite float32 ---
    print("\nConverting to TFLite (float32)...")
    convert_float32(keras_model)
    tflite_f32_preds = run_tflite_inference(TFLITE_FLOAT32_PATH, X_test_scaled, is_int8=False)
    results["tflite_float32"] = evaluate(y_test, tflite_f32_preds)
    results["tflite_float32"]["size_kb"] = round(
        os.path.getsize(TFLITE_FLOAT32_PATH) / 1024, 2
    )
    print(f"  MAE={results['tflite_float32']['mae']:.3f}  R2={results['tflite_float32']['r2']:.3f}"
          f"  size={results['tflite_float32']['size_kb']} KB")

    # --- 3. TFLite int8 (quantized, NPU-ready) ---
    print("\nConverting to TFLite (int8 quantized)...")
    convert_int8(keras_model, X_train_scaled)
    tflite_int8_preds = run_tflite_inference(TFLITE_INT8_PATH, X_test_scaled, is_int8=True)
    results["tflite_int8"] = evaluate(y_test, tflite_int8_preds)
    results["tflite_int8"]["size_kb"] = round(
        os.path.getsize(TFLITE_INT8_PATH) / 1024, 2
    )
    print(f"  MAE={results['tflite_int8']['mae']:.3f}  R2={results['tflite_int8']['r2']:.3f}"
          f"  size={results['tflite_int8']['size_kb']} KB")

    # --- summary ---
    print("\n" + "=" * 60)
    print(f"{'Model':<18}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'Size(KB)':>12}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<18}{m['mae']:>8.3f}{m['rmse']:>8.3f}{m['r2']:>8.3f}{m['size_kb']:>12}")
    print("=" * 60)

    size_reduction = (
        (results["keras_float32"]["size_kb"] - results["tflite_int8"]["size_kb"])
        / results["keras_float32"]["size_kb"]
        * 100
    )
    r2_drop = results["keras_float32"]["r2"] - results["tflite_int8"]["r2"]
    print(f"\nInt8 quantization: {size_reduction:.1f}% size reduction, "
          f"R2 drop of {r2_drop:.4f} vs original Keras model.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "quantization_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved:")
    print(f"  - {TFLITE_FLOAT32_PATH}")
    print(f"  - {TFLITE_INT8_PATH}")
    print(f"  - {RESULTS_DIR}/quantization_comparison.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
