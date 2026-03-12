"""
train.py

CLI training script: loads a CSV of simulated or real driving data,
extracts 12-feature vectors (9 IMU + 3 route-lookahead), trains a Keras MLP,
exports to TensorFlow Lite, and saves the scaler + feature schema.

Usage:
    python -m python.train --input data/sim_trips.csv --out-model models/curve_detector.tflite
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .feature_extractor import (
    FEATURE_NAMES,
    FEATURE_ORDER,
    NUM_FEATURES,
    extract_window_features,
    fit_scaler,
    apply_scaler,
)

# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------

LABEL_MAP = {0: "safe", 1: "mild", 2: "urgent", 3: "hectic"}


def label_window(severity: float) -> int:
    """
    Auto-label a window based on the severity_proxy (v²/r).
    Thresholds are tuned for synthetic data at realistic road speeds.
    """
    if severity < 3.0:
        return 0  # safe
    if severity < 5.0:
        return 1  # mild
    if severity < 8.0:
        return 2  # urgent
    return 3       # hectic


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def process_data(csv_path: str):
    """
    Load a trip CSV, extract 12-feature windows, and assign labels.

    The CSV may include pre-computed lookahead columns
    (lookahead_severity, map_curve_radius, distance_to_next_curve) from
    simulate.py. If they are absent the features default to safe values.

    Returns:
        X_df: DataFrame of shape (n_windows, 12)
        y:    numpy array of integer labels
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    has_lookahead = all(
        c in df.columns
        for c in ("lookahead_severity", "map_curve_radius", "distance_to_next_curve")
    )
    if not has_lookahead:
        print("  No lookahead columns found — using defaults (0, 9999, 999).")
        df["lookahead_severity"]     = 0.0
        df["map_curve_radius"]       = 9999.0
        df["distance_to_next_curve"] = 999.0

    # Sliding window parameters
    window_size = 60   # 3 s at 20 Hz
    step_size   = 10   # 0.5 s step

    X_rows: list = []
    y_labels: list = []

    print(f"Extracting 12-feature windows (size={window_size}, step={step_size})...")
    for i in range(0, len(df) - window_size, step_size):
        window = df.iloc[i: i + window_size]

        # IMU + GPS curvature
        feats = extract_window_features(
            window,
            # Pass lookahead from the *end* of the window (most recent reading)
            lookahead_severity=float(window["lookahead_severity"].iloc[-1]),
            map_curve_radius=float(window["map_curve_radius"].iloc[-1]),
            distance_to_next_curve=float(window["distance_to_next_curve"].iloc[-1]),
        )

        severity = feats["severity_proxy"]   # label from current-position proxy
        label = label_window(severity)

        X_rows.append(list(feats.values()))
        y_labels.append(label)

    X_df = pd.DataFrame(X_rows, columns=FEATURE_NAMES)
    y = np.array(y_labels)

    print(f"  Generated {len(X_df)} windows with {NUM_FEATURES} features each.")
    return X_df, y


def train_stats(y: np.ndarray) -> None:
    classes, counts = np.unique(y, return_counts=True)
    print("Label distribution:", dict(zip([LABEL_MAP[c] for c in classes], counts)))


# ---------------------------------------------------------------------------
# Train & export
# ---------------------------------------------------------------------------

def train_and_export(
    input_csv: str,
    out_model_path: str,
    quick: bool = False,
) -> None:
    # 1. Prepare data
    X_df, y = process_data(input_csv)
    train_stats(y)

    # 2. Fit + save scaler
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    scaler_path = os.path.join(os.path.dirname(out_model_path), "scaler.pkl")
    scaler = fit_scaler(X_df, scaler_path)
    X_scaled = apply_scaler(X_df, scaler)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 3. Build Keras MLP
    # Architecture: Input(12) → Dense(64, relu) → Dropout(0.3)
    #             → Dense(32, relu) → Dense(4, softmax)
    model = keras.Sequential([
        keras.layers.Input(shape=(NUM_FEATURES,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(4, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    epochs = 2 if quick else 50
    print(f"\nTraining for {epochs} epochs (quick={quick})...")
    model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True
            )
        ] if not quick else [],
        verbose=1,
    )

    # 4. Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report (12-feature MLP):")
    print(classification_report(
        y_test, y_pred,
        target_names=[LABEL_MAP[i] for i in range(4)],
    ))

    # 5. Export TFLite
    print(f"\nExporting TFLite model → {out_model_path}")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(out_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"  Model size: {len(tflite_model) / 1024:.1f} KB")

    # 6. Save feature schema
    schema_path = os.path.join(os.path.dirname(out_model_path), "feature_schema.json")
    schema = {
        "num_features": NUM_FEATURES,
        "features": FEATURE_NAMES,
        "scaler": "scaler.pkl",
        "label_map": LABEL_MAP,
        "model_input_shape": [1, NUM_FEATURES],
        "model_output_shape": [1, 4],
    }
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"  Schema saved → {schema_path}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export 12-feature curve detector.")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument(
        "--out-model",
        type=str,
        default="models/curve_detector.tflite",
        help="Output TFLite model path",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run 2 epochs only (for CI)",
    )
    args = parser.parse_args()
    train_and_export(args.input, args.out_model, args.quick)
