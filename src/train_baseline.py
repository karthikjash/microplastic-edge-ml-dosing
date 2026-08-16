"""
Baseline Model Training
========================
Trains and compares baseline regressors for predicting microplastic
concentration (mg/L) from the 5 fluorescence-derived features:

    peak_intensity, mean_intensity, rise_time, decay_time, energy

Models trained:
    1. Linear Regression      (sanity baseline)
    2. Random Forest          (non-linear sanity baseline)
    3. Small Feedforward NN   (Keras -> this is the one we'll later
                                quantize / convert for the FRDM-MCXN236
                                eIQ Neutron NPU via TFLite)

Outputs:
    models/linear_regression.joblib
    models/random_forest.joblib
    models/nn_baseline.keras          (if TensorFlow is installed)
    models/feature_scaler.joblib      (StandardScaler used before the NN)
    results/metrics.json
    results/pred_vs_true.png
    results/nn_training_curve.png     (if TensorFlow is installed)

Usage:
    python3 src/train_baseline.py
"""

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "synthetic_fluorescence_dataset.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
RANDOM_STATE = 42

FEATURE_COLS = ["peak_intensity", "mean_intensity", "rise_time", "decay_time", "energy"]
TARGET_COL = "concentration_mgl"


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run src/generate_dataset.py first."
        )
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_linear(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds)
    return model, preds, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds)
    return model, preds, metrics


def train_nn(X_train, y_train, X_test, y_test):
    """Small feedforward NN. Kept intentionally tiny (<10K params) since
    this is the architecture we'll later quantize (TFLite int8) for the
    FRDM-MCXN236 eIQ Neutron NPU."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("\n[!] TensorFlow not installed - skipping NN training.")
        print("    Install with: pip install tensorflow --break-system-packages")
        return None, None, None, None

    tf.random.set_seed(RANDOM_STATE)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1, activation="linear"),
        ],
        name="baseline_concentration_nn",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss="mse", metrics=["mae"])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )

    preds = model.predict(X_test, verbose=0).flatten()
    metrics = evaluate(y_test, preds)
    n_params = model.count_params()
    metrics["n_params"] = int(n_params)
    return model, preds, metrics, history


def plot_predictions(results, y_test, out_path):
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, results.items()):
        ax.scatter(y_test, preds, s=8, alpha=0.4)
        lims = [0, max(y_test.max(), preds.max()) * 1.05]
        ax.plot(lims, lims, "r--", linewidth=1, label="ideal")
        ax.set_xlabel("true concentration (mg/L)")
        ax.set_ylabel("predicted concentration (mg/L)")
        ax.set_title(name)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_nn_history(history, out_path):
    plt.figure(figsize=(6, 4.5))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("NN baseline training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading dataset...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    all_metrics = {}
    pred_results = {}

    # --- 1. Linear Regression ---
    print("\nTraining Linear Regression...")
    lin_model, lin_preds, lin_metrics = train_linear(X_train, y_train, X_test, y_test)
    all_metrics["linear_regression"] = lin_metrics
    pred_results["Linear Regression"] = lin_preds
    joblib.dump(lin_model, os.path.join(MODELS_DIR, "linear_regression.joblib"))
    print(f"  MAE={lin_metrics['mae']:.3f}  RMSE={lin_metrics['rmse']:.3f}  R2={lin_metrics['r2']:.3f}")

    # --- 2. Random Forest ---
    print("\nTraining Random Forest...")
    rf_model, rf_preds, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    all_metrics["random_forest"] = rf_metrics
    pred_results["Random Forest"] = rf_preds
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.joblib"))
    print(f"  MAE={rf_metrics['mae']:.3f}  RMSE={rf_metrics['rmse']:.3f}  R2={rf_metrics['r2']:.3f}")

    # --- 3. Small NN (scaled features) ---
    print("\nTraining baseline NN (Keras)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "feature_scaler.joblib"))

    nn_out = train_nn(X_train_scaled, y_train, X_test_scaled, y_test)
    if nn_out[0] is not None:
        nn_model, nn_preds, nn_metrics, history = nn_out
        all_metrics["nn_baseline"] = nn_metrics
        pred_results["NN Baseline"] = nn_preds
        nn_model.save(os.path.join(MODELS_DIR, "nn_baseline.keras"))
        print(
            f"  MAE={nn_metrics['mae']:.3f}  RMSE={nn_metrics['rmse']:.3f}  "
            f"R2={nn_metrics['r2']:.3f}  params={nn_metrics['n_params']}"
        )
        plot_nn_history(history, os.path.join(RESULTS_DIR, "nn_training_curve.png"))

    # --- save metrics + plots ---
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    plot_predictions(pred_results, y_test, os.path.join(RESULTS_DIR, "pred_vs_true.png"))

    print("\nSaved:")
    print(f"  - {RESULTS_DIR}/metrics.json")
    print(f"  - {RESULTS_DIR}/pred_vs_true.png")
    print(f"  - {MODELS_DIR}/*.joblib" + (" and nn_baseline.keras" if nn_out[0] is not None else ""))
    print("\nDone.")


if __name__ == "__main__":
    main()