package com.example.driverwarning

import kotlin.math.abs

/**
 * Merges the 9 IMU-derived features from [FeatureExtractor] with the
 * 3 route-lookahead features from [RouteCurvePredictor] to produce a
 * 12-element feature vector for [TFLiteModelRunner].
 *
 * Feature layout (must match Python FEATURE_NAMES order exactly):
 *
 *  [0]  current_speed
 *  [1]  mean_speed
 *  [2]  speed_std
 *  [3]  gyro_z_mean
 *  [4]  gyro_z_max
 *  [5]  acc_y_mean
 *  [6]  acc_y_max
 *  [7]  curve_radius
 *  [8]  severity_proxy
 *  [9]  lookahead_severity      ← route-ahead
 *  [10] map_curve_radius        ← route-ahead
 *  [11] distance_to_next_curve  ← route-ahead
 */
class SensorFusionEngine {

    companion object {
        const val NUM_IMU_FEATURES = 9
        const val NUM_TOTAL_FEATURES = 12
    }

    // -------------------------------------------------------------------------
    // Fusion
    // -------------------------------------------------------------------------

    /**
     * Concatenate [imuFeatures] (size 9) with [lookahead] to produce a
     * 12-element FloatArray ready for normalization and TFLite inference.
     *
     * Falls back to safe defaults if [lookahead] is null (pure IMU mode).
     */
    fun fuse(
        imuFeatures: FloatArray,
        lookahead: LookaheadResult?,
    ): FloatArray {
        require(imuFeatures.size == NUM_IMU_FEATURES) {
            "Expected $NUM_IMU_FEATURES IMU features, got ${imuFeatures.size}"
        }

        val fused = FloatArray(NUM_TOTAL_FEATURES)
        imuFeatures.copyInto(fused, destinationOffset = 0)

        if (lookahead != null) {
            fused[9]  = lookahead.lookaheadSeverity
            fused[10] = lookahead.mapCurveRadius
            fused[11] = lookahead.distanceToNextCurve
        } else {
            // Safe defaults: no curve ahead
            fused[9]  = 0.0f                                    // lookahead_severity
            fused[10] = RouteCurvePredictor.STRAIGHT_RADIUS     // map_curve_radius
            fused[11] = RouteCurvePredictor.NO_CURVE_DIST       // distance_to_next_curve
        }

        return fused
    }

    // -------------------------------------------------------------------------
    // Normalization  (StandardScaler from Python fit)
    // -------------------------------------------------------------------------

    /**
     * Normalize all 12 features using the mean/scale from the retrained
     * Python StandardScaler.
     *
     * Update [ScalerParams12] after each retraining run by reading the
     * printed scaler values from train.py.
     */
    fun normalizeAll(features: FloatArray): FloatArray {
        require(features.size == NUM_TOTAL_FEATURES) {
            "Expected $NUM_TOTAL_FEATURES features for normalization, got ${features.size}"
        }
        return FloatArray(NUM_TOTAL_FEATURES) { i ->
            (features[i] - ScalerParams12.mean[i]) / ScalerParams12.scale[i]
        }
    }

    // -------------------------------------------------------------------------
    // Fusion quality diagnostics
    // -------------------------------------------------------------------------

    /**
     * Returns a human-readable string describing the lookahead state.
     * Use in Logcat or UI for debugging.
     */
    fun describeState(lookahead: LookaheadResult?): String {
        if (lookahead == null) return "IMU-only (no route)"
        val sev = when {
            lookahead.lookaheadSeverity < 1.0f -> "safe"
            lookahead.lookaheadSeverity < 3.0f -> "mild"
            lookahead.lookaheadSeverity < 6.0f -> "urgent"
            else                               -> "hectic"
        }
        val distStr = if (lookahead.distanceToNextCurve >= RouteCurvePredictor.NO_CURVE_DIST - 1f)
            "no curve ahead" else "curve in ${lookahead.distanceToNextCurve.toInt()} m"
        return "Map: $sev @ r=${lookahead.mapCurveRadius.toInt()} m | $distStr"
    }
}

// =============================================================================
// Scaler parameters for all 12 features
// =============================================================================

/**
 * StandardScaler parameters for the 12-feature model.
 *
 * Features 0-8 are carried from [ScalerParams] (re-fitted on new data).
 * Features 9-11 are the route-lookahead features with approximate initial
 * values based on the synthetic training distribution.
 *
 * IMPORTANT: After retraining the Python model, update ALL 12 mean/scale
 * values by reading the printed scaler stats from train.py or from
 * scaler.pkl via:
 *
 *   import joblib, numpy as np
 *   sc = joblib.load("models/scaler.pkl")
 *   print(sc.mean_)
 *   print(sc.scale_)
 */
object ScalerParams12 {

    val mean = floatArrayOf(
        // --- IMU features (0-8) --- from retrained scaler.pkl ---
        15.2539683f,   // [0]  current_speed
        15.2539683f,   // [1]  mean_speed
        0.5954401f,    // [2]  speed_std
        0.1541937f,    // [3]  gyro_z_mean
        0.2912044f,    // [4]  gyro_z_max
        2.6617357f,    // [5]  acc_y_mean
        4.9118627f,    // [6]  acc_y_max
        6683.4618402f, // [7]  curve_radius
        2.2706252f,    // [8]  severity_proxy
        // --- Route-lookahead features (9-11) --- from retrained scaler.pkl ---
        8.2873301f,    // [9]  lookahead_severity
        716.0450593f,  // [10] map_curve_radius
        104.3730117f,  // [11] distance_to_next_curve
    )

    val scale = floatArrayOf(
        // --- IMU features (0-8) ---
        2.2464664f,    // [0]  current_speed
        1.8758554f,    // [1]  mean_speed
        1.0950449f,    // [2]  speed_std
        0.2064617f,    // [3]  gyro_z_mean
        0.3193558f,    // [4]  gyro_z_max
        4.3343240f,    // [5]  acc_y_mean
        6.9171400f,    // [6]  acc_y_max
        4688.9060585f, // [7]  curve_radius
        4.4915750f,    // [8]  severity_proxy
        // --- Route-lookahead features (9-11) ---
        2.3735767f,    // [9]  lookahead_severity
        2509.8392148f, // [10] map_curve_radius
        251.0580711f,  // [11] distance_to_next_curve
    )
}
