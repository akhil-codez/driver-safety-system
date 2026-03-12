package com.example.driverwarning

import kotlin.math.*

/**
 * Data class representing a single sensor sample.
 */
data class SensorSample(
    val timestamp: Long,
    val lat: Double,
    val lon: Double,
    val speed: Float,
    val heading: Float,
    val accX: Float,
    val accY: Float,
    val accZ: Float,
    val gyroX: Float,
    val gyroY: Float,
    val gyroZ: Float
)

/**
 * Scaler parameters extracted from Python StandardScaler.
 * 
 * These values are from the trained scaler.pkl file.
 */
/**
 * StandardScaler parameters for the 9 raw IMU features.
 * These are used by [FeatureExtractor.normalize] for IMU-only normalisation.
 * The full 12-feature normalisation (IMU + lookahead) is handled by
 * [SensorFusionEngine.normalizeAll] using [ScalerParams12].
 */
object ScalerParams9 {
    val mean = floatArrayOf(
        14.8717949f,   // current_speed
        14.8717949f,   // mean_speed
        0.565802792f,  // speed_std
        0.132884451f,  // gyro_z_mean
        0.252287531f,  // gyro_z_max
        2.93112313f,   // acc_y_mean
        4.60716055f,   // acc_y_max
        4591.55258f,   // curve_radius
        3.47271410f    // severity_proxy
    )

    val scale = floatArrayOf(
        2.11441314f,   // current_speed
        1.79563476f,   // mean_speed
        0.973361201f,  // speed_std
        1.65431942f,   // gyro_z_mean
        2.45094490f,   // gyro_z_max
        3.38653469f,   // acc_y_mean
        4.89897919f,   // acc_y_max
        4166.67773f,   // curve_radius
        4.02332926f    // severity_proxy
    )
}

/**
 * Feature extraction from sensor windows.
 * 
 * CRITICAL: Feature computation must match Python exactly!
 */
class FeatureExtractor {
    
    /**
     * Extract 9 raw IMU features from a sensor window.
     *
     * These 9 features are subsequently merged with 3 route-lookahead features
     * by [SensorFusionEngine.fuse] to produce the 12-element TFLite input.
     *
     * @param window List of sensor samples (typically 30 samples for 3 s at 10 Hz)
     * @return FloatArray of 9 raw (unnormalized) IMU features
     */
    fun extractFeatures(window: List<SensorSample>): FloatArray {
        require(window.isNotEmpty()) { "Window cannot be empty" }
        
        val features = FloatArray(9)
        
        // 1. current_speed - last speed in window
        features[0] = window.last().speed
        
        // 2. mean_speed - average speed
        features[1] = window.map { it.speed }.average().toFloat()
        
        // 3. speed_std - standard deviation of speed
        val meanSpeed = features[1]
        val variance = window.map { (it.speed - meanSpeed).pow(2) }.average()
        features[2] = sqrt(variance).toFloat()
        
        // 4. gyro_z_mean - average yaw rate
        features[3] = window.map { it.gyroZ }.average().toFloat()
        
        // 5. gyro_z_max - max absolute yaw rate
        features[4] = window.map { abs(it.gyroZ) }.maxOrNull() ?: 0f
        
        // 6. acc_y_mean - average lateral acceleration
        features[5] = window.map { it.accY }.average().toFloat()
        
        // 7. acc_y_max - max absolute lateral acceleration
        features[6] = window.map { abs(it.accY) }.maxOrNull() ?: 0f
        
        // 8. curve_radius - from GPS polyline
        val gpsPoints = window.takeLast(5).map { Pair(it.lat, it.lon) }
        features[7] = computeCurveRadius(gpsPoints)
        
        // 9. severity_proxy - speed^2 / radius
        val radius = maxOf(features[7], 1f)
        features[8] = features[0].pow(2) / radius
        
        return features
    }
    
    /**
     * Compute radius of curvature from GPS points using Menger curvature.
     */
    private fun computeCurveRadius(points: List<Pair<Double, Double>>): Float {
        if (points.size < 3) return 999999f
        
        // Use last 3 points for triangle
        val p1 = points[points.size - 3]
        val p2 = points[points.size - 2]
        val p3 = points[points.size - 1]
        
        // Compute side lengths using haversine
        val a = haversineDistance(p1, p2)
        val b = haversineDistance(p2, p3)
        val c = haversineDistance(p3, p1)
        
        // Avoid degenerate triangles
        if (a < 0.1 || b < 0.1 || c < 0.1) return 999999f
        
        // Heron's formula for area
        val s = (a + b + c) / 2
        val areaSquared = s * (s - a) * (s - b) * (s - c)
        
        if (areaSquared <= 0) return 999999f
        
        val area = sqrt(areaSquared)
        
        // Menger curvature: κ = 4 * Area / (a * b * c)
        val curvature = (4 * area) / (a * b * c)
        
        if (curvature < 1e-6) return 999999f
        
        val radius = 1.0 / curvature
        return radius.coerceIn(1.0, 999999.0).toFloat()
    }
    
    /**
     * Haversine distance between two GPS points.
     */
    private fun haversineDistance(p1: Pair<Double, Double>, p2: Pair<Double, Double>): Double {
        val R = 6371000.0 // Earth radius in meters
        
        val lat1 = Math.toRadians(p1.first)
        val lat2 = Math.toRadians(p2.first)
        val dLat = Math.toRadians(p2.first - p1.first)
        val dLon = Math.toRadians(p2.second - p1.second)
        
        val a = sin(dLat / 2).pow(2) + cos(lat1) * cos(lat2) * sin(dLon / 2).pow(2)
        val c = 2 * asin(sqrt(a))
        
        return R * c
    }
    
    /**
     * Normalize the 9 raw IMU features using [ScalerParams9].
     *
     * For full 12-feature normalization used at inference time, see
     * [SensorFusionEngine.normalizeAll].
     *
     * Formula: (x - mean) / scale
     */
    fun normalize(features: FloatArray): FloatArray {
        require(features.size == 9) { "Expected 9 IMU features, got ${features.size}" }
        return FloatArray(9) { i ->
            (features[i] - ScalerParams9.mean[i]) / ScalerParams9.scale[i]
        }
    }
}
