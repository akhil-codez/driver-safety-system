package com.example.driverwarning

/**
 * Result of scanning ahead along the planned route.
 *
 * These three values are appended to the 9 IMU features to create the
 * 12-feature vector fed into TFLite.
 */
data class LookaheadResult(
    /** Max theoretical lateral accel (v²/r) in next 500 m at reference speed. */
    val lookaheadSeverity: Float,
    /** Minimum curve radius found in lookahead window (m). 9999 = straight. */
    val mapCurveRadius: Float,
    /** Distance to the nearest sharp curve (radius < 200 m) in metres. 999 = none found. */
    val distanceToNextCurve: Float,
)
