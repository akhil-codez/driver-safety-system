# Feature Schema Documentation

This document describes the exact features used by the Driver Warning System model, their computation methods, and ordering.

## Feature Overview

The model uses **9 features** extracted from 3-second sliding windows of sensor data. Features are computed at 1-second intervals (overlapping windows).

## Feature List (Exact Order)

The features **must** be provided in this exact order for TFLite inference:

| # | Feature Name | Unit | Description |
|---|--------------|------|-------------|
| 1 | `current_speed` | m/s | Instantaneous speed at end of window |
| 2 | `mean_speed` | m/s | Average speed over 3-second window |
| 3 | `speed_std` | m/s | Standard deviation of speed in window |
| 4 | `gyro_z_mean` | deg/s | Average yaw rate (rotation around vertical axis) |
| 5 | `gyro_z_max` | deg/s | Maximum absolute yaw rate in window |
| 6 | `acc_y_mean` | m/s² | Average lateral acceleration (perpendicular to motion) |
| 7 | `acc_y_max` | m/s² | Maximum absolute lateral acceleration |
| 8 | `curve_radius` | m | Radius of curvature from GPS polyline |
| 9 | `severity_proxy` | m/s² | Combined metric: `speed² / radius` |

## Detailed Feature Descriptions

### 1. current_speed
- **Computation**: Last speed value in the 3-second window
- **Physical meaning**: Vehicle's instantaneous speed
- **Typical range**: 0-30 m/s (0-108 km/h for urban/highway driving)
- **Importance**: High - directly affects curve danger

### 2. mean_speed
- **Computation**: `mean(speed[t-3s:t])`
- **Physical meaning**: Average speed over recent history
- **Typical range**: 0-30 m/s
- **Importance**: Medium - context for speed changes

### 3. speed_std
- **Computation**: `std(speed[t-3s:t])`
- **Physical meaning**: Speed variability (acceleration/deceleration)
- **Typical range**: 0-5 m/s
- **Importance**: Medium - indicates unstable driving

### 4. gyro_z_mean
- **Computation**: `mean(gyro_z[t-3s:t])`
- **Physical meaning**: Average yaw rate (turning rate)
- **Typical range**: -30 to +30 deg/s (negative = left turn, positive = right turn)
- **Importance**: **Very High** - primary indicator of turning

### 5. gyro_z_max
- **Computation**: `max(abs(gyro_z[t-3s:t]))`
- **Physical meaning**: Peak turning rate in window
- **Typical range**: 0-50 deg/s
- **Importance**: **Very High** - captures sharp turns

### 6. acc_y_mean
- **Computation**: `mean(acc_y[t-3s:t])`
- **Physical meaning**: Average lateral (sideways) acceleration
- **Typical range**: -5 to +5 m/s² (negative = left, positive = right)
- **Importance**: **Very High** - centripetal force in curves

### 7. acc_y_max
- **Computation**: `max(abs(acc_y[t-3s:t]))`
- **Physical meaning**: Peak lateral acceleration
- **Typical range**: 0-8 m/s²
- **Importance**: **Very High** - indicates curve sharpness

### 8. curve_radius
- **Computation**: Menger curvature from GPS points using formula:
  ```
  κ = 4 * Area(triangle) / (a * b * c)
  radius = 1 / κ
  ```
  where a, b, c are triangle side lengths from 3 consecutive GPS points
- **Physical meaning**: Radius of the curve being navigated
- **Typical range**: 
  - Straight: >500 m
  - Gentle curve: 200-500 m
  - Sharp curve: 100-200 m
  - Very sharp: <100 m
- **Importance**: **Critical** - used for auto-labeling and severity assessment

### 9. severity_proxy
- **Computation**: `current_speed² / max(curve_radius, 1)`
- **Physical meaning**: Approximates lateral acceleration (v²/r formula)
- **Typical range**: 0-10 m/s²
- **Importance**: **Critical** - combines speed and curvature into danger metric

## Auto-Labeling Thresholds

Windows are automatically labeled based on `curve_radius`:

| Label | Severity | Radius Range | Description |
|-------|----------|--------------|-------------|
| 0 | Safe | > 500 m | Straight or very gentle curve |
| 1 | Mild | 200-500 m | Gentle curve, normal caution |
| 2 | Urgent | 100-200 m | Sharp curve, reduce speed |
| 3 | Hectic | < 100 m | Very sharp curve, high danger |

## Sensor Requirements

### GPS
- **Frequency**: 1 Hz minimum (10 Hz recommended for better curvature estimation)
- **Accuracy**: 5-10 meters typical
- **Outputs**: latitude, longitude, speed, heading

### IMU (Accelerometer + Gyroscope)
- **Frequency**: 10 Hz minimum (50 Hz recommended)
- **Accelerometer range**: ±8g (±78.4 m/s²)
- **Gyroscope range**: ±250 deg/s
- **Coordinate system**: 
  - X: forward (longitudinal)
  - Y: right (lateral)
  - Z: down (vertical)

## Preprocessing Steps

1. **Collect sensor data** at 10 Hz
2. **Create 3-second windows** (30 samples) with 1-second step (10 samples)
3. **Extract features** using `feature_extractor.py`
4. **Normalize** using fitted StandardScaler:
   ```python
   X_scaled = (X - scaler.mean_) / scaler.scale_
   ```
5. **Feed to TFLite** as float32 array of shape `[1, 9]`

## Scaler Parameters

The StandardScaler is fitted on training data and saved to `models/scaler.pkl`. 

For Android integration, extract scaler parameters:
```python
import pickle
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

mean = scaler.mean_  # shape: (9,)
scale = scaler.scale_  # shape: (9,)
```

Then in Kotlin:
```kotlin
val mean = floatArrayOf(...)  // from scaler.mean_
val scale = floatArrayOf(...) // from scaler.scale_

fun normalize(features: FloatArray): FloatArray {
    return features.mapIndexed { i, x -> (x - mean[i]) / scale[i] }.toFloatArray()
}
```

## Model Output

The TFLite model outputs a **softmax probability distribution** over 4 classes:

```
Output shape: [1, 4]
Output[0] = [prob_safe, prob_mild, prob_urgent, prob_hectic]
```

Example:
```
[0.85, 0.10, 0.04, 0.01]  → Safe (85% confidence)
[0.05, 0.15, 0.60, 0.20]  → Urgent (60% confidence)
```

## Consistency Requirements

⚠️ **Critical**: Python and Android **must** compute features identically.

- Use same formulas
- Same window size (3 seconds)
- Same sample rate (10 Hz)
- Same feature order
- Same scaler parameters
- Same coordinate system for IMU

Any mismatch will cause incorrect predictions!
