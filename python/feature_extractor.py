"""
feature_extractor.py

Extracts a 12-feature vector from a 3-second sensor window combined with
route-lookahead data from route_curve_analyzer.

Feature layout (12 total):
  [0]  current_speed       — current speed (m/s)
  [1]  mean_speed          — window-average speed
  [2]  speed_std           — speed variability
  [3]  gyro_z_mean         — avg absolute yaw rate (deg/s)
  [4]  gyro_z_max          — peak yaw rate
  [5]  acc_y_mean          — avg lateral acceleration (m/s²)
  [6]  acc_y_max           — peak lateral acceleration
  [7]  curve_radius        — GPS-derived radius for current window (m)
  [8]  severity_proxy      — v²/r for current position
  --- map lookahead features ---
  [9]  lookahead_severity  — worst v²/r in next 500 m at ref speed
  [10] map_curve_radius    — min radius found ahead (m)
  [11] distance_to_next_curve — metres to nearest sharp turn (999 = none)
"""

import pandas as pd
import numpy as np
import joblib
from collections import OrderedDict
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature names (authoritative order — must match Android FeatureExtractor.kt)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # IMU / GPS window features
    "current_speed",
    "mean_speed",
    "speed_std",
    "gyro_z_mean",
    "gyro_z_max",
    "acc_y_mean",
    "acc_y_max",
    "curve_radius",
    "severity_proxy",
    # Route lookahead features
    "lookahead_severity",
    "map_curve_radius",
    "distance_to_next_curve",
]

# Legacy alias used by some existing callers
FEATURE_ORDER = FEATURE_NAMES

NUM_FEATURES = len(FEATURE_NAMES)  # 12


# ---------------------------------------------------------------------------
# Default lookahead values (safe / no-route fallbacks)
# ---------------------------------------------------------------------------

DEFAULT_LOOKAHEAD_SEVERITY: float = 0.0
DEFAULT_MAP_CURVE_RADIUS: float = 9999.0
DEFAULT_DISTANCE_TO_NEXT_CURVE: float = 999.0


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_severity(speed: float, radius: float) -> float:
    """Theoretical lateral acceleration v²/r (m/s²)."""
    if radius <= 0 or np.isinf(radius):
        return 0.0
    return (speed ** 2) / radius


def sliding_windows(
    df: pd.DataFrame,
    window_sec: float = 3.0,
    step_sec: float = 1.0,
    sample_rate_hz: float = 10.0,
) -> List[pd.DataFrame]:
    """
    Slice a dataframe into overlapping fixed-length windows.

    Args:
        df:              Input dataframe with at least the sensor columns.
        window_sec:      Window length in seconds.
        step_sec:        Step size between windows in seconds.
        sample_rate_hz:  Samples per second.

    Returns:
        List of dataframe slices (each has ``window_size`` rows).
    """
    window_size = int(window_sec * sample_rate_hz)
    step_size = int(step_sec * sample_rate_hz)
    windows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        windows.append(df.iloc[start: start + window_size])
    return windows


def extract_window_features(
    window_df: pd.DataFrame,
    curve_radius: Optional[float] = None,
    lookahead_severity: float = DEFAULT_LOOKAHEAD_SEVERITY,
    map_curve_radius: float = DEFAULT_MAP_CURVE_RADIUS,
    distance_to_next_curve: float = DEFAULT_DISTANCE_TO_NEXT_CURVE,
) -> OrderedDict:
    """
    Extract all 12 features from a sensor window.

    IMU columns required: ``speed``, ``gyro_z``, ``acc_y``.
    GPS columns optional (used for curve_radius if not provided): ``lat``, ``lon``.

    The three lookahead parameters default to safe values when no route
    polyline is available (pure IMU-only mode).

    Args:
        window_df:              DataFrame slice for the 3-second window.
        curve_radius:           Override curve radius (m); if None, computed
                                from GPS lat/lon in ``window_df``.
        lookahead_severity:     Route-ahead feature: max v²/r ahead.
        map_curve_radius:       Route-ahead feature: min curve radius (m).
        distance_to_next_curve: Route-ahead feature: metres to next curve.

    Returns:
        OrderedDict with 12 features in FEATURE_NAMES order.
    """
    features: OrderedDict = OrderedDict()

    if window_df.empty:
        for name in FEATURE_NAMES[:9]:
            features[name] = 0.0
        features["lookahead_severity"] = lookahead_severity
        features["map_curve_radius"] = map_curve_radius
        features["distance_to_next_curve"] = distance_to_next_curve
        return features

    # ---- Speed features ----
    features["current_speed"] = float(window_df["speed"].iloc[-1])
    features["mean_speed"] = float(window_df["speed"].mean())
    features["speed_std"] = float(window_df["speed"].std()) if len(window_df) > 1 else 0.0

    # ---- Gyro features (yaw) ----
    features["gyro_z_mean"] = float(window_df["gyro_z"].abs().mean())
    features["gyro_z_max"] = float(window_df["gyro_z"].abs().max())

    # ---- Accel features (lateral) ----
    features["acc_y_mean"] = float(window_df["acc_y"].abs().mean())
    features["acc_y_max"] = float(window_df["acc_y"].abs().max())

    # ---- GPS-derived curve radius ----
    if curve_radius is not None:
        features["curve_radius"] = float(curve_radius) if curve_radius > 0 else 9999.0
    elif "lat" in window_df.columns and "lon" in window_df.columns:
        coords = list(zip(window_df["lat"].tail(5), window_df["lon"].tail(5)))
        features["curve_radius"] = _compute_radius_from_coords(coords)
    else:
        features["curve_radius"] = 9999.0

    # ---- Severity proxy (current position) ----
    features["severity_proxy"] = compute_severity(
        features["mean_speed"], features["curve_radius"]
    )

    # ---- Route lookahead features ----
    features["lookahead_severity"] = float(lookahead_severity)
    features["map_curve_radius"] = float(map_curve_radius) if map_curve_radius > 0 else 9999.0
    features["distance_to_next_curve"] = float(distance_to_next_curve)

    return features


def _compute_radius_from_coords(coords: list) -> float:
    """Menger-curvature-based radius from up to 5 GPS points."""
    import math

    def _hav(lat1, lon1, lat2, lon2):
        R = 6_371_000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    if len(coords) < 3:
        return 9999.0
    p1, p2, p3 = coords[-3], coords[-2], coords[-1]
    a = _hav(*p1, *p2)
    b = _hav(*p2, *p3)
    c = _hav(*p3, *p1)
    if a < 0.1 or b < 0.1 or c < 0.1:
        return 9999.0
    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 0:
        return 9999.0
    area = math.sqrt(area_sq)
    k = (4 * area) / (a * b * c)
    if k < 1e-7:
        return 9999.0
    return min(1.0 / k, 9999.0)


# ---------------------------------------------------------------------------
# Batch feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_windows(
    windows: List[pd.DataFrame],
    lookahead_list: Optional[list] = None,
) -> pd.DataFrame:
    """
    Extract features from a list of windows and return a DataFrame.

    Args:
        windows:        List of window DataFrames.
        lookahead_list: Optional list of LookaheadResult (or dict) objects,
                        one per window. If None, defaults are used.

    Returns:
        DataFrame with 12 columns (FEATURE_NAMES) and one row per window.
    """
    rows = []
    for i, window in enumerate(windows):
        if lookahead_list is not None and i < len(lookahead_list):
            la = lookahead_list[i]
            # Support both dataclass-style and dict-style
            if hasattr(la, "lookahead_severity"):
                feats = extract_window_features(
                    window,
                    lookahead_severity=la.lookahead_severity,
                    map_curve_radius=la.map_curve_radius,
                    distance_to_next_curve=la.distance_to_next_curve,
                )
            else:
                feats = extract_window_features(
                    window,
                    lookahead_severity=la.get("lookahead_severity", DEFAULT_LOOKAHEAD_SEVERITY),
                    map_curve_radius=la.get("map_curve_radius", DEFAULT_MAP_CURVE_RADIUS),
                    distance_to_next_curve=la.get("distance_to_next_curve", DEFAULT_DISTANCE_TO_NEXT_CURVE),
                )
        else:
            feats = extract_window_features(window)
        rows.append(list(feats.values()))
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------

def fit_scaler(df: pd.DataFrame, output_path: Optional[str] = None) -> StandardScaler:
    """
    Fit a StandardScaler on feature columns and optionally save it.

    Args:
        df:          DataFrame with columns matching FEATURE_NAMES.
        output_path: If given, saves scaler to this path with joblib.

    Returns:
        Fitted StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_NAMES])
    if output_path:
        joblib.dump(scaler, output_path)
        print(f"Scaler saved to {output_path}")
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Apply a fitted scaler object to a numpy array or DataFrame.

    Args:
        X:      Input array/DataFrame of shape (n_samples, 12).
        scaler: Fitted StandardScaler.

    Returns:
        Scaled numpy array.
    """
    if isinstance(X, pd.DataFrame):
        return scaler.transform(X[FEATURE_NAMES])
    return scaler.transform(X)
