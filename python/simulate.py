"""
simulate.py

Physics-based synthetic driving trip generator. Produces a CSV of sensor
readings (speed, acc_y, gyro_z, GPS) with realistic noise.

Now also generates the three route-lookahead features by treating the
simulated trajectory as the "planned route" and scanning ahead from each
window's position — giving the training data realistic map-fusion signals.
"""

import pandas as pd
import numpy as np
import math
import argparse
import random
from datetime import datetime, timedelta
from typing import List, Tuple

# Constants
G = 9.81


# ---------------------------------------------------------------------------
# Low-level segment generator
# ---------------------------------------------------------------------------

def generate_segment(
    start_lat: float,
    start_lon: float,
    start_heading: float,
    speed_mps: float,
    duration_sec: float,
    turn_rate_deg_per_sec: float,
    dt: float = 0.05,
) -> dict:
    """
    Generate a segment of driving data at a fixed speed and turning rate.

    acc_y is computed from centripetal force: a = v * ω (rad/s).
    Gaussian noise is added to all sensor readings to mimic real smartphone
    sensors.
    """
    t = np.arange(0, duration_sec, dt)

    lats, lons, headings, speeds = [], [], [], []
    acc_xs, acc_ys, acc_zs = [], [], []
    gyro_xs, gyro_ys, gyro_zs = [], [], []

    curr_lat = start_lat
    curr_lon = start_lon
    curr_heading = start_heading

    # Angular velocity in rad/s
    omega_rad = math.radians(turn_rate_deg_per_sec)

    for _ in t:
        R_earth = 6_371_000.0
        dx = speed_mps * math.cos(math.radians(curr_heading)) * dt  # North component
        dy = speed_mps * math.sin(math.radians(curr_heading)) * dt  # East component

        d_lat = math.degrees(dx / R_earth)
        d_lon = math.degrees(dy / (R_earth * math.cos(math.radians(curr_lat))))

        curr_lat += d_lat
        curr_lon += d_lon
        curr_heading = (curr_heading + turn_rate_deg_per_sec * dt) % 360

        lats.append(curr_lat)
        lons.append(curr_lon)
        headings.append(curr_heading)
        speeds.append(speed_mps)

        gyro_z = omega_rad   # yaw rate
        acc_y = speed_mps * omega_rad  # centripetal: v·ω

        # Sensor noise
        acc_xs.append(np.random.normal(0, 0.1))
        acc_ys.append(acc_y + np.random.normal(0, 0.2))
        acc_zs.append(G + np.random.normal(0, 0.1))
        gyro_xs.append(np.random.normal(0, 0.01))
        gyro_ys.append(np.random.normal(0, 0.01))
        gyro_zs.append(gyro_z + np.random.normal(0, 0.02))

    return dict(
        lat=lats, lon=lons, heading=headings, speed=speeds,
        acc_x=acc_xs, acc_y=acc_ys, acc_z=acc_zs,
        gyro_x=gyro_xs, gyro_y=gyro_ys, gyro_z=gyro_zs,
    )


# ---------------------------------------------------------------------------
# Full trip generator
# ---------------------------------------------------------------------------

def generate_trip(out_file: str) -> None:
    """
    Simulate a full trip with multiple straight/curve segments, compute
    route-lookahead features for each row, and save to CSV.
    """
    print(f"Generating trip to {out_file}...")

    columns = [
        "utc_ts", "lat", "lon", "speed", "heading",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
    ]
    data = {k: [] for k in columns}

    start_time = datetime.now()
    curr_lat, curr_lon = 37.7749, -122.4194  # San Francisco seed
    curr_heading = 0.0

    # (label, duration_sec, turn_rate_deg_per_sec)
    segments = [
        ("straight",           10, 0),
        ("curve_right_mild",    5, 10),
        ("straight",            5, 0),
        ("curve_left_sharp",    4, -25),
        ("straight",           10, 0),
        ("curve_right_urgent",  3, 40),
        ("straight",            8, 0),
        ("curve_left_mild",     6, -12),
        ("straight",            7, 0),
        ("curve_right_hectic",  3, 60),
        ("straight",            5, 0),
    ]

    dt = 0.05  # 20 Hz
    current_ts = start_time

    for label, duration, turn_rate in segments:
        speed = 15.0            # default ~54 km/h
        if "sharp"  in label: speed = 10.0
        if "urgent" in label: speed = 20.0
        if "hectic" in label: speed = 22.0

        seg = generate_segment(
            curr_lat, curr_lon, curr_heading, speed, duration, turn_rate, dt
        )

        N = len(seg["lat"])
        for i in range(N):
            data["utc_ts"].append(
                (current_ts + timedelta(seconds=i * dt)).timestamp()
            )
            for k in seg:
                data[k].append(seg[k][i])

        curr_lat = seg["lat"][-1]
        curr_lon = seg["lon"][-1]
        curr_heading = seg["heading"][-1]
        current_ts += timedelta(seconds=N * dt)

    df = pd.DataFrame(data)

    # -----------------------------------------------------------------------
    # Inject route-lookahead features
    # Build the full trajectory as the "planned route" polyline, then scan
    # ahead from each point so training rows contain realistic map signals.
    # -----------------------------------------------------------------------
    try:
        from python.route_curve_analyzer import scan_ahead_batch, LookaheadResult
        from python.route_geometry import resample_polyline

        # Use the GPS trace as the route polyline (resampled to 10 m)
        raw_coords = list(zip(df["lat"], df["lon"]))
        # Deduplicate consecutive identical points
        route = [raw_coords[0]]
        for p in raw_coords[1:]:
            if p != route[-1]:
                route.append(p)
        route_resampled = resample_polyline(route, spacing_m=10.0)

        query_pts = raw_coords
        results: List[LookaheadResult] = scan_ahead_batch(
            route_resampled, query_pts, lookahead_m=500.0
        )

        df["lookahead_severity"]     = [r.lookahead_severity     for r in results]
        df["map_curve_radius"]       = [r.map_curve_radius       for r in results]
        df["distance_to_next_curve"] = [r.distance_to_next_curve for r in results]

        print("  Lookahead features injected successfully.")

    except Exception as exc:  # noqa: BLE001
        print(f"  Warning: could not compute lookahead features ({exc}). "
              "Using defaults (0, 9999, 999).")
        df["lookahead_severity"]     = 0.0
        df["map_curve_radius"]       = 9999.0
        df["distance_to_next_curve"] = 999.0

    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} samples to {out_file}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic driving data.")
    parser.add_argument("--out", type=str, required=True, help="Output CSV file")
    args = parser.parse_args()
    generate_trip(args.out)
