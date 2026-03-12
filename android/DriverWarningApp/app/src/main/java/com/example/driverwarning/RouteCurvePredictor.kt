package com.example.driverwarning

import com.google.android.gms.maps.model.LatLng
import kotlin.math.*

/**
 * Walks [lookaheadM] metres ahead along the planned route polyline from
 * the vehicle's current position and computes [LookaheadResult].
 *
 * This mirrors route_curve_analyzer.py exactly so training and inference
 * use the same logic.
 */
class RouteCurvePredictor {

    companion object {
        /** Reference speed for severity proxy (≈ 50 km/h). */
        const val REFERENCE_SPEED_MPS = 14.0f
        /** How far ahead to scan (metres). */
        const val LOOKAHEAD_M = 500.0f
        /** Radius below this is considered a "sharp curve" (metres). */
        const val CURVE_THRESHOLD_M = 200.0f
        /** Sentinel: effectively straight road. */
        const val STRAIGHT_RADIUS = 9999.0f
        /** Sentinel: no curve found in lookahead. */
        const val NO_CURVE_DIST = 999.0f
    }

    // -------------------------------------------------------------------------
    // Main API
    // -------------------------------------------------------------------------

    /**
     * Compute lookahead features from the route polyline.
     *
     * @param polyline   Resampled route waypoints from [RouteManager].
     * @param currentLat Current vehicle latitude.
     * @param currentLon Current vehicle longitude.
     * @param lookaheadM Distance (m) to scan ahead.
     * @return [LookaheadResult] with three route-derived features,
     *         or safe defaults if the polyline is too short.
     */
    fun scanAhead(
        polyline: List<LatLng>,
        currentLat: Double,
        currentLon: Double,
        lookaheadM: Float = LOOKAHEAD_M,
    ): LookaheadResult {
        if (polyline.size < 3) {
            return LookaheadResult(0f, STRAIGHT_RADIUS, NO_CURVE_DIST)
        }

        val startIdx = nearestIndex(polyline, currentLat, currentLon)

        // Collect the lookahead window
        val window = mutableListOf<LatLng>()
        var accumulated = 0.0

        for (i in startIdx until polyline.size) {
            window.add(polyline[i])
            if (i > startIdx) {
                val prev = polyline[i - 1]
                accumulated += haversineM(
                    prev.latitude, prev.longitude,
                    polyline[i].latitude, polyline[i].longitude,
                )
            }
            if (accumulated >= lookaheadM) break
        }

        if (window.size < 3) {
            return LookaheadResult(0f, STRAIGHT_RADIUS, NO_CURVE_DIST)
        }

        // Radius at each window point
        val radii = computeRadii(window)

        // Running distances within window
        val runningDist = FloatArray(window.size)
        for (i in 1 until window.size) {
            runningDist[i] = runningDist[i - 1] + haversineM(
                window[i - 1].latitude, window[i - 1].longitude,
                window[i].latitude,     window[i].longitude,
            ).toFloat()
        }

        // map_curve_radius: minimum radius in window
        val minRadius = radii.minOrNull() ?: STRAIGHT_RADIUS
        val mapCurveRadius = maxOf(minRadius, 1.0f)

        // lookahead_severity: v²/r at reference speed
        val lookaheadSeverity = (REFERENCE_SPEED_MPS * REFERENCE_SPEED_MPS) / mapCurveRadius

        // distance_to_next_curve: distance to nearest radius < threshold
        var distanceToNextCurve = NO_CURVE_DIST
        for (i in radii.indices) {
            if (radii[i] < CURVE_THRESHOLD_M) {
                distanceToNextCurve = runningDist[i]
                break
            }
        }

        return LookaheadResult(lookaheadSeverity, mapCurveRadius, distanceToNextCurve)
    }

    // -------------------------------------------------------------------------
    // Geometry 
    // -------------------------------------------------------------------------

    /**
     * Compute the curve radius at each point using Menger curvature
     * (circumradius of the triangle formed by three consecutive points).
     */
    fun computeRadii(points: List<LatLng>): FloatArray {
        val n = points.size
        if (n < 3) return FloatArray(n) { STRAIGHT_RADIUS }

        val radii = FloatArray(n) { STRAIGHT_RADIUS }
        for (i in 1 until n - 1) {
            radii[i] = mengerRadius(points[i - 1], points[i], points[i + 1])
        }
        radii[0] = radii[1]
        radii[n - 1] = radii[n - 2]
        return radii
    }

    /**
     * Circumradius from three GPS points using Menger curvature formula.
     * Returns STRAIGHT_RADIUS for collinear or near-collinear points.
     */
    fun mengerRadius(p1: LatLng, p2: LatLng, p3: LatLng): Float {
        val a = haversineM(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
        val b = haversineM(p2.latitude, p2.longitude, p3.latitude, p3.longitude)
        val c = haversineM(p3.latitude, p3.longitude, p1.latitude, p1.longitude)

        if (a < 0.5 || b < 0.5 || c < 0.5) return STRAIGHT_RADIUS

        val s = (a + b + c) / 2.0
        val areaSq = s * (s - a) * (s - b) * (s - c)
        if (areaSq <= 0.0) return STRAIGHT_RADIUS

        val area = sqrt(areaSq)
        val curvature = (4.0 * area) / (a * b * c)
        if (curvature < 1e-7) return STRAIGHT_RADIUS

        return minOf((1.0 / curvature).toFloat(), STRAIGHT_RADIUS)
    }

    /** Find the index of the polyline point nearest to (lat, lon). */
    fun nearestIndex(polyline: List<LatLng>, lat: Double, lon: Double): Int {
        var bestIdx = 0
        var bestDist = Double.MAX_VALUE
        polyline.forEachIndexed { i, pt ->
            val d = haversineM(lat, lon, pt.latitude, pt.longitude)
            if (d < bestDist) { bestDist = d; bestIdx = i }
        }
        return bestIdx
    }

    /** Great-circle distance in metres (Haversine). */
    fun haversineM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val R = 6_371_000.0
        val phi1 = Math.toRadians(lat1)
        val phi2 = Math.toRadians(lat2)
        val dPhi = Math.toRadians(lat2 - lat1)
        val dLam = Math.toRadians(lon2 - lon1)
        val a = sin(dPhi / 2).pow(2) + cos(phi1) * cos(phi2) * sin(dLam / 2).pow(2)
        return R * 2 * asin(sqrt(a))
    }
}
