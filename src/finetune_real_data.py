"""
Real-Data Fine-Tuning (Transfer Learning)
============================================
Takes the synthetic-trained baseline NN and fine-tunes it on a small
batch of REAL photodiode readings, to close the sim-to-real gap before
a live demo. Uses a low learning rate and few epochs so it corrects
toward real-world data without catastrophically forgetting the broader
patterns learned from the larger synthetic dataset.

Expects a CSV of real readings with the same 5 feature columns as the
synthetic dataset, plus the true concentration:

    peak_intensity,mean_intensity,rise_time,decay_time,energy,concentration_mgl

to calibrate with the coherent real time data acquired from the initial testing, 
re run this script. 

Outputs:
    models/nn_finetuned.keras
    models/nn_finetuned_int8.tflite
    results/finetune_comparison.json

Usage:
    python3 src/finetune_real_data.py --csv data/real_readings.csv
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "nn_baseline.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.joblib")
FINETUNED_MODEL_PATH = os.path.join(MODELS_DIR, "nn_finetuned.keras")
FINETUNED_TFLITE_PATH = os.path.join(MODELS_DIR, "nn_finetuned_int8.tflite")

FEATURE_COLS = ["peak_intensity", "mean_intensity", "rise_time", "decay_time", "energy"]
TARGET_COL = "concentration_mgl"


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def finetune(csv_path, epochs=30, learning_rate=0.0005, val_split=0.2):
    print(f"Loading real data from {csv_path}...")
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    print(f"Loaded {len(X)} real samples.")

    if len(X) < 5:
        print("[!] Warning: very few real samples. Fine-tuning may overfit badly.")

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X).astype(np.float32)

    print("Loading synthetic-trained baseline model...")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    # Evaluate BEFORE fine-tuning (synthetic-only model on real data)
    pre_preds = model.predict(X_scaled, verbose=0).flatten()
    pre_metrics = evaluate(y, pre_preds)
    print(f"\nBEFORE fine-tuning (synthetic-only model on real data):")
    print(f"  MAE={pre_metrics['mae']:.3f}  RMSE={pre_metrics['rmse']:.3f}  R2={pre_metrics['r2']:.3f}")

    # Recompile with a low learning rate for gentle fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    n_val = max(1, int(len(X_scaled) * val_split)) if len(X_scaled) >= 5 else 0
    fit_kwargs = dict(epochs=epochs, batch_size=min(8, len(X_scaled)), verbose=0)
    if n_val > 0:
        fit_kwargs["validation_split"] = val_split

    print(f"\nFine-tuning for {epochs} epochs at lr={learning_rate}...")
    model.fit(X_scaled, y, **fit_kwargs)

    # Evaluate AFTER fine-tuning
    post_preds = model.predict(X_scaled, verbose=0).flatten()
    post_metrics = evaluate(y, post_preds)
    print(f"\nAFTER fine-tuning (on real data, same samples used to tune - optimistic estimate):")
    print(f"  MAE={post_metrics['mae']:.3f}  RMSE={post_metrics['rmse']:.3f}  R2={post_metrics['r2']:.3f}")

    model.save(FINETUNED_MODEL_PATH)
    print(f"\nSaved fine-tuned model -> {FINETUNED_MODEL_PATH}")

    # Re-quantize to int8 using the real data as representative dataset
    print("\nRe-quantizing fine-tuned model to int8...")

    def representative_dataset():
        for i in range(len(X_scaled)):
            yield [X_scaled[i : i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open(FINETUNED_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"Saved fine-tuned int8 model -> {FINETUNED_TFLITE_PATH}")

    # Save comparison
    os.makedirs(RESULTS_DIR, exist_ok=True)
    comparison = {
        "n_real_samples": len(X),
        "before_finetuning": pre_metrics,
        "after_finetuning": post_metrics,
    }
    with open(os.path.join(RESULTS_DIR, "finetune_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved comparison -> {RESULTS_DIR}/finetune_comparison.json")

    print("\n" + "=" * 60)
    print("NOTE: 'after' metrics above are on the same data used to fine-tune,")
    print("so they're optimistic. For a fair number, hold out a few real")
    print("samples and evaluate on those separately if you can spare them.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune baseline model on real photodiode data")
    parser.add_argument("--csv", type=str, required=True, help="Path to real readings CSV")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()

    finetune(args.csv, epochs=args.epochs, learning_rate=args.lr)


if __name__ == "__main__":
    main()