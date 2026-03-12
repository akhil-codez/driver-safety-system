"""
route_curve_analyzer.py

Pre-scans the planned route geometry ahead of the vehicle's current position
and extracts three lookahead features that are merged with live IMU data:

  lookahead_severity    – max theoretical lateral accel (v²/r) in next 500 m
  map_curve_radius      – minimum curve radius (m) found in next 500 m
  distance_to_next_curve– metres to the nearest radius < 200 m curve

These features give the ML model map-awareness so it can warn the driver
*before* the IMU sensors react to the curve.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Public data type
# ---------------------------------------------------------------------------

@dataclass
class LookaheadResult:
    """Three route-derived features for one inference step."""
    lookahead_severity: float       # m/s² — v²/r at reference_speed_mps over next 500 m
    map_curve_radius: float         # m    — min radius in lookahead window (9999 = straight)
    distance_to_next_curve: float   # m    — distance to nearest sharp curve (999 = none)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_SPEED_MPS: float = 14.0   # ~50 km/h — speed used for severity proxy
LOOKAHEAD_M: float = 500.0          # metres to scan ahead
CURVE_THRESHOLD_M: float = 200.0    # radius < this is considered a "curve"
STRAIGHT_RADIUS: float = 9999.0     # sentinel for effectively straight road
NO_CURVE_DIST: float = 999.0        # sentinel when no curve found in window


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _menger_radius(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
) -> float:
    """
    Circumradius of the triangle formed by three GPS points (Menger curvature).
    Returns STRAIGHT_RADIUS when the three points are collinear or too close.
    """
    a = _haversine_m(*p1, *p2)
    b = _haversine_m(*p2, *p3)
    c = _haversine_m(*p3, *p1)

    if a < 0.5 or b < 0.5 or c < 0.5:
        return STRAIGHT_RADIUS

    # Heron's formula for triangle area
    s = (a + b + c) / 2.0
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 0.0:
        return STRAIGHT_RADIUS

    area = math.sqrt(area_sq)
    curvature = (4.0 * area) / (a * b * c)

    if curvature < 1e-7:
        return STRAIGHT_RADIUS

    return min(1.0 / curvature, STRAIGHT_RADIUS)


def _nearest_index(
    polyline: List[Tuple[float, float]],
    lat: float,
    lon: float,
) -> int:
    """Return the index of the polyline point closest to (lat, lon)."""
    best_idx = 0
    best_dist = float("inf")
    for i, (plat, plon) in enumerate(polyline):
        d = _haversine_m(lat, lon, plat, plon)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Per-point radius computation for a polyline
# ---------------------------------------------------------------------------

def compute_radii(
    polyline: List[Tuple[float, float]],
) -> List[float]:
    """
    Compute the curve radius at every point in a polyline using the
    Menger curvature of three consecutive points.

    Returns a list of the same length as ``polyline``.
    """
    n = len(polyline)
    if n < 3:
        return [STRAIGHT_RADIUS] * n

    radii: List[float] = [STRAIGHT_RADIUS] * n
    for i in range(1, n - 1):
        radii[i] = _menger_radius(polyline[i - 1], polyline[i], polyline[i + 1])

    # Endpoints inherit their neighbour's value
    radii[0] = radii[1]
    radii[-1] = radii[-2]
    return radii


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def scan_ahead(
    polyline: List[Tuple[float, float]],
    current_lat: float,
    current_lon: float,
    lookahead_m: float = LOOKAHEAD_M,
    reference_speed_mps: float = REFERENCE_SPEED_MPS,
    curve_threshold_m: float = CURVE_THRESHOLD_M,
) -> LookaheadResult:
    """
    Walk ``lookahead_m`` metres ahead along ``polyline`` from the vehicle's
    current position and compute the three lookahead features.

    Args:
        polyline:            Resampled (lat, lon) route waypoints.
        current_lat:         Vehicle latitude.
        current_lon:         Vehicle longitude.
        lookahead_m:         How far ahead to scan (metres).
        reference_speed_mps: Speed used for computing the severity proxy (v²/r).
        curve_threshold_m:   Radius below which a section is a "curve".

    Returns:
        LookaheadResult with lookahead_severity, map_curve_radius,
        distance_to_next_curve.
    """
    if len(polyline) < 3:
        return LookaheadResult(
            lookahead_severity=0.0,
            map_curve_radius=STRAIGHT_RADIUS,
            distance_to_next_curve=NO_CURVE_DIST,
        )

    start_idx = _nearest_index(polyline, current_lat, current_lon)

    # Collect polyline points within lookahead_m of current position
    window: List[Tuple[float, float]] = []
    accumulated_dist = 0.0

    for i in range(start_idx, len(polyline)):
        window.append(polyline[i])

        if i > start_idx:
            accumulated_dist += _haversine_m(
                *polyline[i - 1], *polyline[i]
            )

        if accumulated_dist >= lookahead_m:
            break

    if len(window) < 3:
        return LookaheadResult(
            lookahead_severity=0.0,
            map_curve_radius=STRAIGHT_RADIUS,
            distance_to_next_curve=NO_CURVE_DIST,
        )

    # Radius at each window point
    radii = compute_radii(window)

    # Compute running distances along the window for distance_to_next_curve
    running_dist = [0.0] * len(window)
    for i in range(1, len(window)):
        running_dist[i] = running_dist[i - 1] + _haversine_m(
            *window[i - 1], *window[i]
        )

    # Feature: map_curve_radius — minimum radius in lookahead window
    min_radius = min(radii)
    map_curve_radius = max(min_radius, 1.0)  # clip to 1 m floor

    # Feature: lookahead_severity — worst-case v²/r (at reference speed)
    lookahead_severity = (reference_speed_mps ** 2) / map_curve_radius

    # Feature: distance_to_next_curve
    distance_to_next_curve = NO_CURVE_DIST
    for i, r in enumerate(radii):
        if r < curve_threshold_m:
            distance_to_next_curve = running_dist[i]
            break

    return LookaheadResult(
        lookahead_severity=lookahead_severity,
        map_curve_radius=map_curve_radius,
        distance_to_next_curve=distance_to_next_curve,
    )


# ---------------------------------------------------------------------------
# Batch variant (for simulation / training data generation)
# ---------------------------------------------------------------------------

def scan_ahead_batch(
    polyline: List[Tuple[float, float]],
    query_points: List[Tuple[float, float]],
    lookahead_m: float = LOOKAHEAD_M,
    reference_speed_mps: float = REFERENCE_SPEED_MPS,
) -> List[LookaheadResult]:
    """
    Efficiently compute scan_ahead for a list of query points on the
    same polyline. Used during training data generation.

    Args:
        polyline:      Resampled route polyline.
        query_points:  List of (lat, lon) positions to scan from.
        lookahead_m:   Lookahead distance in metres.
        reference_speed_mps: Speed for severity proxy.

    Returns:
        List of LookaheadResult, one per query point.
    """
    # Pre-compute all radii for the whole polyline once
    all_radii = compute_radii(polyline)

    # Pre-compute cumulative distance along polyline
    cum_dist = [0.0] * len(polyline)
    for i in range(1, len(polyline)):
        cum_dist[i] = cum_dist[i - 1] + _haversine_m(
            *polyline[i - 1], *polyline[i]
        )

    results: List[LookaheadResult] = []
    for (lat, lon) in query_points:
        start_idx = _nearest_index(polyline, lat, lon)
        start_dist = cum_dist[start_idx]
        end_dist = start_dist + lookahead_m

        # Collect indices in lookahead window
        window_radii: List[float] = []
        window_dists: List[float] = []
        for i in range(start_idx, len(polyline)):
            if cum_dist[i] > end_dist:
                break
            window_radii.append(all_radii[i])
            window_dists.append(cum_dist[i] - start_dist)

        if not window_radii:
            results.append(LookaheadResult(0.0, STRAIGHT_RADIUS, NO_CURVE_DIST))
            continue

        min_r = max(min(window_radii), 1.0)
        severity = (reference_speed_mps ** 2) / min_r
        dist_to_curve = NO_CURVE_DIST
        for r, d in zip(window_radii, window_dists):
            if r < CURVE_THRESHOLD_M:
                dist_to_curve = d
                break

        results.append(LookaheadResult(severity, min_r, dist_to_curve))

    return results


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick self-test with a synthetic L-shaped route
    straight = [(37.7749 + i * 0.0001, -122.4194) for i in range(60)]    # heading north
    turn = [(37.7749 + 0.006, -122.4194 + i * 0.0001) for i in range(1, 40)]    # sharp left
    route = straight + turn

    result_before = scan_ahead(route, 37.7749, -122.4194)
    result_on_curve = scan_ahead(route, 37.7749 + 0.004, -122.4194)

    print("=== Route Curve Analyzer Self-Test ===")
    print(f"[Start of straight] lookahead_severity={result_before.lookahead_severity:.3f}  "
          f"map_curve_radius={result_before.map_curve_radius:.1f}m  "
          f"dist_to_curve={result_before.distance_to_next_curve:.1f}m")
    print(f"[Near bend]         lookahead_severity={result_on_curve.lookahead_severity:.3f}  "
          f"map_curve_radius={result_on_curve.map_curve_radius:.1f}m  "
          f"dist_to_curve={result_on_curve.distance_to_next_curve:.1f}m")
