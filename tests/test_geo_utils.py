"""
Unit tests for geo_utils module.
"""

import math
import pytest
from python.geo_utils import haversine_m, bearing, curvature_from_polyline


def test_haversine_known_distance():
    """Test haversine with known distance (NYC to LA)."""
    # NYC: 40.7128° N, 74.0060° W
    # LA: 34.0522° N, 118.2437° W
    # Expected distance: ~3936 km
    distance = haversine_m(40.7128, -74.0060, 34.0522, -118.2437)
    
    # Allow 1% error
    expected = 3936000  # meters
    assert abs(distance - expected) < expected * 0.01, f"Expected ~{expected}m, got {distance}m"


def test_haversine_same_point():
    """Test haversine with same point (should be 0)."""
    distance = haversine_m(40.0, -74.0, 40.0, -74.0)
    assert distance < 1.0, f"Expected ~0m, got {distance}m"


def test_bearing_cardinal_directions():
    """Test bearing for cardinal directions."""
    # North: same longitude, increasing latitude
    b_north = bearing(40.0, -74.0, 41.0, -74.0)
    assert abs(b_north - 0) < 1, f"North bearing should be ~0°, got {b_north}°"
    
    # East: same latitude, increasing longitude
    b_east = bearing(40.0, -74.0, 40.0, -73.0)
    assert abs(b_east - 90) < 1, f"East bearing should be ~90°, got {b_east}°"
    
    # South: same longitude, decreasing latitude
    b_south = bearing(40.0, -74.0, 39.0, -74.0)
    assert abs(b_south - 180) < 1, f"South bearing should be ~180°, got {b_south}°"
    
    # West: same latitude, decreasing longitude
    b_west = bearing(40.0, -74.0, 40.0, -75.0)
    assert abs(b_west - 270) < 1, f"West bearing should be ~270°, got {b_west}°"


def test_curvature_straight_line():
    """Test curvature for a straight line (should have near-zero curvature)."""
    # Three points in a straight line (heading north)
    points = [
        (0.0, 0.0),
        (0.001, 0.0),
        (0.002, 0.0)
    ]
    
    curvatures = curvature_from_polyline(points)
    
    assert len(curvatures) == 3
    # Middle curvature should be very small (straight road → k ≈ 0)
    for k in curvatures:
        assert k < 0.01, f"Straight line should have near-zero curvature, got {k}"


def test_curvature_triangle():
    """Test curvature with a simple triangle (non-collinear points)."""
    # Three points forming a triangle
    points = [
        (0.0, 0.0),
        (0.001, 0.0),
        (0.0005, 0.001)
    ]
    
    curvatures = curvature_from_polyline(points)
    
    assert len(curvatures) == 3
    # Middle point should have non-zero curvature
    assert curvatures[1] > 0, f"Triangle middle point should have positive curvature, got {curvatures[1]}"


def test_curvature_insufficient_points():
    """Test curvature with insufficient points (< 3)."""
    points = [(0.0, 0.0), (0.001, 0.0)]
    
    curvatures = curvature_from_polyline(points)
    
    assert len(curvatures) == 2
    # Should return 0 curvature (no turn info without 3 points)
    for k in curvatures:
        assert k == 0.0, f"Expected 0.0 curvature, got {k}"


def test_curvature_sharp_curve():
    """Test curvature with a sharp curve."""
    # Points forming a tight 90-degree corner
    points = [
        (0.0, 0.0),
        (0.0001, 0.0),
        (0.0001, 0.0001),
        (0.0, 0.0001)
    ]
    
    curvatures = curvature_from_polyline(points)
    
    assert len(curvatures) == 4
    # Middle points should have significantly positive curvature (sharp turn)
    assert curvatures[1] > 0.001, f"Sharp curve should have positive curvature, got {curvatures[1]}"
    assert curvatures[2] > 0.001, f"Sharp curve should have positive curvature, got {curvatures[2]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
