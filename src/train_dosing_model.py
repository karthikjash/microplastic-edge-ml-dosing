"""
Model B Training - Substance X Dose Prediction
==================================================
Trains a small regression model predicting minimum effective dose
(mg/L of Substance X) from contamination level (mg/L, Model A's output).

Same NPU-size constraint as Model A: kept small (<10K params) since it
will eventually also be quantized (TFLite int8) for the FRDM-MCXN236.

Outputs:
    models/dosing_model.keras
    models/dosing_scaler.joblib
    results/dosing_metrics.json
    results/dosing_pred_vs_true.png

"""

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "synthetic_dosing_dataset.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
RANDOM_STATE = 42

FEATURE_COL = "contamination_mgl"
TARGET_COL = "dose_mgl"


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading dosing dataset...")
    df = pd.read_csv(DATA_PATH)
    X = df[[FEATURE_COL]].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "dosing_scaler.joblib"))

    tf.random.set_seed(RANDOM_STATE)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(12, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ],
        name="dosing_model",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae"])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    print("\nTraining dosing model...")
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.15,
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )

    preds = model.predict(X_test_scaled, verbose=0).flatten()
    metrics = evaluate(y_test, preds)
    metrics["n_params"] = int(model.count_params())
    print(f"\nMAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.3f}  params={metrics['n_params']}")

    model.save(os.path.join(MODELS_DIR, "dosing_model.keras"))

    with open(os.path.join(RESULTS_DIR, "dosing_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # plot: dose vs contamination, predicted vs true, across the full range
    plt.figure(figsize=(6.5, 5))
    order = np.argsort(X_test.flatten())
    plt.scatter(X_test.flatten(), y_test, s=10, alpha=0.4, label="true dose")
    plt.scatter(X_test.flatten()[order], preds[order], s=10, alpha=0.4, label="predicted dose")
    plt.xlabel("contamination (mg/L)")
    plt.ylabel("Substance X dose (mg/L)")
    plt.title("Model B: dose prediction vs contamination")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "dosing_pred_vs_true.png"), dpi=130)

    print(f"\nSaved:")
    print(f"  - {MODELS_DIR}/dosing_model.keras")
    print(f"  - {MODELS_DIR}/dosing_scaler.joblib")
    print(f"  - {RESULTS_DIR}/dosing_metrics.json")
    print(f"  - {RESULTS_DIR}/dosing_pred_vs_true.png")
    print("\nDone.")


if __name__ == "__main__":
    main()