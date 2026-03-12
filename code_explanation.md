# Driver Safety Project - Codebase Explanation

This document explains the purpose, key functions, and outputs of each file in your current **Preprocessing Stage** repository.

## 1. Core Logic Files

### `python/simulate.py`
**Use:**
Generates synthetic driving data. Since real-world "dangerous driving" data is hard to collect safely, this script simulates a vehicle moving along a defined path (straight, mild curve, sharp curve) and calculates physics-based sensor readings.

**Key Functions:**
*   `generate_segment(...)`: Mathematical heart of the simulator. It calculates:
    *   **Position**: Updates Latitude/Longitude based on speed and heading.
    *   **Centripetal Acceleration**: Uses the formula $a = v^2 / r$ to calculate the lateral G-force you would feel in a curve.
    *   **Noise Injection**: Adds random "jitter" to the data to mimic real, imperfect smartphone sensor data.
*   `generate_trip(...)`: Stitches together multiple segments (e.g., Straight -> Right Turn -> Straight -> Left Sharp Turn) to create a full driving session.

**Output:**
*   `data/live_test.csv`: A single trip file used for quick testing and demos.

### `generate_large_data.py`
**Use:**
Scale up! While `simulate.py` creates one short trip, this script wraps it to generate **hundreds** of diverse driving scenarios. This is what we use to create the "Big Data" needed to train the Neural Network accurately.

**Key Functions:**
*   `generate_large_dataset(out_file, num_trips)`:
    *   **Loop**: Runs a loop 50-100 times to create many unique trips.
    *   **Scenarios**: Defines specific "recipes" for driving (e.g., "Safe Driving", "Hectic Cornering") to ensure the dataset has a good balance of easy and hard examples.
    *   **Randomization**: Adds variance to speed and turn angles so the model doesn't overfit to just one specific pattern.

**Output:**
*   `data/large_training_data.csv`: A massive CSV file containing hours of simulated driving data, ready for `train.py`.

---

### `python/geo_utils.py`
**Use:**
A utility library for handling GPS mathematics. This is essential for calculating "Ground Truth" about the road geometry from raw latitude/longitude points.

**Key Functions:**
*   `haversine_m(lat1, lon1, lat2, lon2)`: Calculates the exact distance in meters between two GPS points on the Earth's sphere.
*   `bearing(...)`: Calculates the compass direction (0-360°) from one point to another.
*   `curvature_from_polyline(coords)`: The most critical function. It looks at three consecutive GPS points to estimate the "Curvature" (sharpness) of the road at that spot using the change in bearing over distance ($k = d\theta / ds$).

**Output:**
*   Returns calculated values (distances, angles, curvature lists) used by other scripts to label data.

---

### `python/feature_extractor.py`
**Use:**
Prepares raw data for the AI. Raw sensor data is too noisy and granular (100s of numbers per second). This script groups data into "Windows" (e.g., 3 seconds of driving) and extracts meaningful "Features" that describe *events*.

**Key Functions:**
*   `extract_window_features(window_df, ...)`: Takes a slice of data and calculates statistics:
    *   **Mean/Max Speed**: How fast were they going?
    *   **Max Gyro Z**: What was the peak rotation rate?
    *   **Severity Proxy**: A synthetic metric combining speed and curve radius to estimate danger level.
*   `fit_scaler(...)` / `apply_scaler(...)`: Standardizes data (scales values to be between -1 and 1). Neural Networks learn much faster when data is "Normalized" like this.

**Output:**
*   Produces a clean "Feature Vector" (a list of summarized numbers) for each 3-second window, ready to be fed into a Machine Learning model.

---

### `visualize_data.py`
**Use:**
The "Proof of Concept" tool. It runs the simulation and plots the results to visually verify that the math works.

**Key Functions:**
*   `create_visualization()`:
    1.  Calls `simulate.py` to get fresh data.
    2.  Uses `matplotlib` to draw two graphs:
        *   **Map View**: The path of the vehicle.
        *   **Sensor View**: Overlays Lateral Acceleration and Gyroscope data to show spikes during curves.

**Output:**
*   `data_preview.png`: An image file showing the graphs. This is what you show in your presentation to illustrate your data.

---

### `python/show_data.py`
**Use:**
A helper for your Live Demo. It prints CSV files in a pretty, readable table format in the terminal so you don't have to open Excel/Notepad.

**Key Functions:**
*   `show_head(file_path)`: Reads the CSV using pandas and prints selected columns (Speed, Accel, Lat, Lon) neatly.

**Output:**
*   Text output in the terminal showing the first 10 rows of your dataset.

---

## 2. Test Files (Validation)

### `tests/test_geo_utils.py`
**Use:**
Ensures your math is correct. It runs automated "Unit Tests" on the functions in `geo_utils.py`.

**Key Functions:**
*   `test_haversine_known_distance()`: Checks if the distance calculation matches the known distance between NYC and LA.
*   `test_bearing_cardinal_directions()`: Verifies that North is 0°, East is 90°, etc.
*   `test_curvature_straight_line()`: Confirms that a straight line correctly calculates as having "Infinite Radius" (Zero Curvature).

**Output:**
*   Pass/Fail status. If you run `pytest`, it tells you if your math logic is broken.

### `tests/test_feature_extractor.py`
**Use:**
Checks if the feature extraction logic correctly computes averages, maximums, and standard deviations from sample data windows.

---

## 3. Configuration Files

### `__init__.py`
**Use:**
Empty files that tell Python "Treat this directory as a Package". This allows you to import files from other folders (e.g., `from python.geo_utils import haversine_m`). Without these, your cross-file imports would fail.
