"""
Unit tests for feature_extractor module.
"""

import numpy as np
import pandas as pd
import pytest
from python.feature_extractor import (
    extract_window_features,
    sliding_windows,
    fit_scaler,
    apply_scaler,
    extract_features_from_windows,
    FEATURE_NAMES,
    NUM_FEATURES,
)


def test_feature_count():
    """Test that exactly 12 features are extracted (9 IMU + 3 route-lookahead)."""
    df = pd.DataFrame({
        'speed': [10, 11, 12, 13, 14],
        'gyro_z': [5, 10, 15, 10, 5],
        'acc_y': [1, 2, 3, 2, 1],
        'lat': [0.0, 0.001, 0.002, 0.003, 0.004],
        'lon': [0.0, 0.0, 0.001, 0.001, 0.002]
    })
    
    features = extract_window_features(df)
    
    assert len(features) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(features)}"


def test_feature_order():
    """Test that features are in correct order."""
    df = pd.DataFrame({
        'speed': [10, 11, 12],
        'gyro_z': [5, 10, 15],
        'acc_y': [1, 2, 3],
        'lat': [0.0, 0.001, 0.002],
        'lon': [0.0, 0.0, 0.001]
    })
    
    features = extract_window_features(df)
    
    assert list(features.keys()) == FEATURE_NAMES, \
        f"Feature order mismatch. Expected {FEATURE_NAMES}, got {list(features.keys())}"


def test_window_extraction():
    """Test sliding window extraction."""
    # Create 100 samples at 10 Hz (10 seconds of data)
    df = pd.DataFrame({'speed': range(100)})
    
    # 3-second windows with 1-second step at 10 Hz
    windows = sliding_windows(df, window_sec=3.0, step_sec=1.0, sample_rate_hz=10.0)
    
    # Each window should have 30 samples (3s * 10Hz)
    assert len(windows[0]) == 30, f"Expected 30 samples per window, got {len(windows[0])}"
    
    # Number of windows: (100 - 30) / 10 + 1 = 8
    expected_windows = (100 - 30) // 10 + 1
    assert len(windows) == expected_windows, f"Expected {expected_windows} windows, got {len(windows)}"


def test_window_overlap():
    """Test that windows overlap correctly."""
    df = pd.DataFrame({'speed': range(50)})
    
    windows = sliding_windows(df, window_sec=3.0, step_sec=1.0, sample_rate_hz=10.0)
    
    # First window should start at index 0
    assert windows[0].iloc[0]['speed'] == 0
    
    # Second window should start at index 10 (1 second step at 10 Hz)
    assert windows[1].iloc[0]['speed'] == 10
    
    # Third window should start at index 20
    assert windows[2].iloc[0]['speed'] == 20


def test_feature_extraction_from_windows():
    """Test feature extraction from multiple windows."""
    # Create synthetic data
    n_samples = 100
    df = pd.DataFrame({
        "speed": np.random.uniform(5, 20, n_samples),
        "gyro_z": np.random.uniform(-10, 10, n_samples),
        "acc_y": np.random.uniform(-2, 2, n_samples),
        "lat": np.linspace(0.0, 0.01, n_samples),
        "lon": np.linspace(0.0, 0.01, n_samples),
    })
    
    windows = sliding_windows(df, window_sec=3, step_sec=1, sample_rate_hz=10)
    features_df = extract_features_from_windows(windows[:5])
    
    # Should have 5 rows (one per window)
    assert features_df.shape[0] == 5, f"Expected 5 rows, got {features_df.shape[0]}"
    
    # Should have 12 columns (9 IMU + 3 lookahead features)
    assert features_df.shape[1] == NUM_FEATURES, f"Expected {NUM_FEATURES} columns, got {features_df.shape[1]}"
    
    # Column names should match FEATURE_NAMES
    assert list(features_df.columns) == FEATURE_NAMES


def test_scaler_fit_apply():
    """Test scaler fitting and application."""
    df = pd.DataFrame(np.random.randn(100, NUM_FEATURES), columns=FEATURE_NAMES)
    
    scaler = fit_scaler(df)
    
    assert scaler.mean_.shape == (NUM_FEATURES,), \
        f"Expected mean shape ({NUM_FEATURES},), got {scaler.mean_.shape}"
    assert scaler.scale_.shape == (NUM_FEATURES,), \
        f"Expected scale shape ({NUM_FEATURES},), got {scaler.scale_.shape}"
    
    X = df.values
    X_scaled = apply_scaler(X, scaler)
    
    assert np.abs(X_scaled.mean(axis=0)).max() < 0.1, "Scaled data should have mean ~0"
    assert np.abs(X_scaled.std(axis=0) - 1.0).max() < 0.1, "Scaled data should have std ~1"


def test_feature_values_reasonable():
    """Test that extracted feature values are reasonable."""
    df = pd.DataFrame({
        'speed': [15.0] * 30,  # Constant 15 m/s
        'gyro_z': [10.0] * 30,  # Constant 10 deg/s
        'acc_y': [2.0] * 30,    # Constant 2 m/s^2
        'lat': np.linspace(0.0, 0.003, 30),
        'lon': np.linspace(0.0, 0.003, 30)
    })
    
    features = extract_window_features(df)
    
    # Current speed should be 15
    assert abs(features["current_speed"] - 15.0) < 0.1
    
    # Mean speed should be 15
    assert abs(features["mean_speed"] - 15.0) < 0.1
    
    # Speed std should be ~0 (constant speed)
    assert features["speed_std"] < 0.1
    
    # Gyro_z mean should be 10
    assert abs(features["gyro_z_mean"] - 10.0) < 0.1
    
    # Acc_y mean should be 2
    assert abs(features["acc_y_mean"] - 2.0) < 0.1


def test_empty_window():
    """Test feature extraction with empty window."""
    df = pd.DataFrame({
        'speed': [],
        'gyro_z': [],
        'acc_y': [],
        'lat': [],
        'lon': []
    })
    
    features = extract_window_features(df)
    
    # Should still return 12 features (9 IMU + 3 lookahead defaults)
    assert len(features) == NUM_FEATURES
    
    # Values should be 0 or default
    assert features["current_speed"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
