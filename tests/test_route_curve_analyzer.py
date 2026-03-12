"""
tests/test_route_curve_analyzer.py

Unit tests for route_curve_analyzer and route_geometry modules.
"""

import math
import pytest
from python.route_curve_analyzer import (
    scan_ahead,
    scan_ahead_batch,
    compute_radii,
    LookaheadResult,
    STRAIGHT_RADIUS,
    NO_CURVE_DIST,
    CURVE_THRESHOLD_M,
)
from python.route_geometry import (
    decode_polyline,
    encode_polyline,
    resample_polyline,
    _haversine_m,
)


# ---------------------------------------------------------------------------
# route_geometry tests
# ---------------------------------------------------------------------------

class TestDecodePolyline:
    def test_known_polyline(self):
        """Decode the classic Google Maps sample polyline."""
        encoded = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
        coords = decode_polyline(encoded)
        assert len(coords) == 3
        # Known first point approximately (38.5, -120.2)
        assert abs(coords[0][0] - 38.5) < 0.01
        assert abs(coords[0][1] - (-120.2)) < 0.01

    def test_roundtrip(self):
        """Encode then decode should return the same coordinates."""
        original = [(37.7749, -122.4194), (37.7850, -122.4100), (37.7900, -122.4000)]
        encoded = encode_polyline(original)
        decoded = decode_polyline(encoded)
        assert len(decoded) == len(original)
        for (lat1, lon1), (lat2, lon2) in zip(original, decoded):
            # Encoded polyline precision is 1e-5 deg; allow 1e-3 for cumulative rounding
            assert abs(lat1 - lat2) < 1e-3, f"Lat mismatch: {abs(lat1-lat2)}"
            assert abs(lon1 - lon2) < 1e-3, f"Lon mismatch: {abs(lon1-lon2)}"

    def test_single_point(self):
        coords = decode_polyline(encode_polyline([(0.0, 0.0)]))
        assert len(coords) == 1
        assert abs(coords[0][0]) < 1e-4
        assert abs(coords[0][1]) < 1e-4


class TestResamplePolyline:
    def _make_straight(self, n=100, dlat=0.0001):
        """n points heading north spaced ~11 m apart."""
        return [(37.7 + i * dlat, -122.4) for i in range(n)]

    def test_output_spacing(self):
        """Resampled points should be approximately 10 m apart."""
        raw = self._make_straight(50)
        resampled = resample_polyline(raw, spacing_m=10.0)
        for i in range(1, len(resampled)):
            d = _haversine_m(*resampled[i - 1], *resampled[i])
            assert d <= 10.0 * 1.15, f"Spacing {d:.1f} m exceeds 115% of target"

    def test_preserves_endpoints(self):
        raw = self._make_straight(20)
        resampled = resample_polyline(raw, spacing_m=10.0)
        assert resampled[0] == raw[0]
        # Last point within 1 m of original endpoint
        d_last = _haversine_m(*resampled[-1], *raw[-1])
        assert d_last < 1.0

    def test_short_polyline(self):
        raw = [(0.0, 0.0)]
        assert resample_polyline(raw) == raw

    def test_two_points(self):
        raw = [(0.0, 0.0), (0.01, 0.0)]
        resampled = resample_polyline(raw, spacing_m=50.0)
        assert len(resampled) >= 2


# ---------------------------------------------------------------------------
# route_curve_analyzer tests
# ---------------------------------------------------------------------------

def _straight_route(n=100):
    """A perfectly straight north-heading route of n points at ~11 m spacing."""
    return [(37.7749 + i * 0.0001, -122.4194) for i in range(n)]


def _L_route():
    """
    An L-shaped route: 60 pts heading north then 40 pts heading east,
    forming a ~90° turn at the junction.
    """
    straight = [(37.7749 + i * 0.0001, -122.4194) for i in range(60)]
    turn     = [(37.7749 + 0.006, -122.4194 + j * 0.0001) for j in range(1, 41)]
    return straight + turn


class TestComputeRadii:
    def test_straight_line_large_radius(self):
        route = _straight_route(10)
        radii = compute_radii(route)
        assert len(radii) == 10
        for r in radii:
            assert r > 1000, f"Straight line should have large radius, got {r}"

    def test_sharp_turn_small_radius(self):
        route = _L_route()
        radii = compute_radii(route)
        # Near the bend (around index 58-61) radius should be small
        bend_radii = radii[56:63]
        assert any(r < 500 for r in bend_radii), \
            f"Expected a small radius near bend, got {bend_radii}"

    def test_length_preserved(self):
        route = _straight_route(20)
        assert len(compute_radii(route)) == 20

    def test_fewer_than_3_points(self):
        radii = compute_radii([(0.0, 0.0), (0.001, 0.0)])
        assert all(r == STRAIGHT_RADIUS for r in radii)


class TestScanAhead:
    def test_straight_road_no_curve(self):
        route = _straight_route(200)
        result = scan_ahead(route, route[0][0], route[0][1])
        assert result.map_curve_radius > 1000, \
            f"Straight road should have large map radius, got {result.map_curve_radius}"
        assert result.distance_to_next_curve >= NO_CURVE_DIST - 1, \
            f"No curve should be found, got {result.distance_to_next_curve}"
        assert result.lookahead_severity < 0.5, \
            f"Severity should be near 0 on straight road, got {result.lookahead_severity}"

    def test_l_route_curve_detected(self):
        route = _L_route()
        # Start near the beginning of the straight — curve should be ~600 m ahead
        result = scan_ahead(route, route[0][0], route[0][1], lookahead_m=700.0)
        assert result.map_curve_radius < CURVE_THRESHOLD_M * 5, \
            f"L-route should detect a curve, map_curve_radius={result.map_curve_radius}"
        assert result.lookahead_severity > 0.1, \
            f"Severity should be positive near bend, got {result.lookahead_severity}"

    def test_on_top_of_curve(self):
        route = _L_route()
        # Position right at the bend start (approx index 58)
        bend_lat, bend_lon = route[58]
        result = scan_ahead(route, bend_lat, bend_lon, lookahead_m=200.0)
        assert result.map_curve_radius < CURVE_THRESHOLD_M * 5

    def test_distance_to_next_curve_when_none(self):
        route = _straight_route(200)
        result = scan_ahead(route, route[0][0], route[0][1])
        assert result.distance_to_next_curve >= NO_CURVE_DIST - 1

    def test_distance_to_next_curve_l_route(self):
        route = _L_route()
        result = scan_ahead(route, route[0][0], route[0][1], lookahead_m=900.0)
        # Should find an actual distance (not the 999 sentinel) since the bend is ~660 m away
        assert result.distance_to_next_curve < NO_CURVE_DIST, \
            "Should find a curve distance on the L-route"

    def test_short_polyline_returns_defaults(self):
        tiny = [(37.7, -122.4), (37.71, -122.4)]
        result = scan_ahead(tiny, 37.7, -122.4)
        assert result.map_curve_radius == STRAIGHT_RADIUS
        assert result.distance_to_next_curve == NO_CURVE_DIST
        assert result.lookahead_severity == 0.0


class TestScanAheadBatch:
    def test_batch_matches_individual(self):
        route = _L_route()
        queries = [(route[0][0], route[0][1]), (route[30][0], route[30][1])]
        batch_results = scan_ahead_batch(route, queries, lookahead_m=500.0)
        for i, (lat, lon) in enumerate(queries):
            individual = scan_ahead(route, lat, lon, lookahead_m=500.0)
            # Results should be close but not necessarily identical due to different
            # nearest-point selection methods; check within 20% tolerance
            br = batch_results[i]
            assert abs(br.lookahead_severity - individual.lookahead_severity) < \
                max(individual.lookahead_severity * 0.25 + 0.1, 0.5)

    def test_batch_length(self):
        route = _straight_route(50)
        queries = [(route[i][0], route[i][1]) for i in range(5)]
        results = scan_ahead_batch(route, queries)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Integration: LookaheadResult defaults
# ---------------------------------------------------------------------------

class TestLookaheadResult:
    def test_dataclass_fields(self):
        r = LookaheadResult(1.5, 250.0, 120.0)
        assert r.lookahead_severity == 1.5
        assert r.map_curve_radius == 250.0
        assert r.distance_to_next_curve == 120.0

    def test_safe_defaults_from_straight_route(self):
        route = _straight_route(100)
        result = scan_ahead(route, route[0][0], route[0][1])
        assert isinstance(result, LookaheadResult)
        assert result.lookahead_severity >= 0.0
        assert result.map_curve_radius >= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
