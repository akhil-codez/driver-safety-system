package com.example.driverwarning

import android.util.Log

/**
 * Simple test to verify model and scaler are working with 12-feature fusion.
 */
object ModelTest {
    private const val TAG = "ModelTest"
    
    fun testModel(modelRunner: TFLiteModelRunner, sensorFusionEngine: SensorFusionEngine) {
        Log.d(TAG, "=== Testing 12-Feature Model ===")
        
        // 1. Mock 9 IMU features (safe)
        val safeImu = floatArrayOf(
            15.0f, 15.0f, 0.5f,  // speed
            0.1f, 0.2f,          // gyro
            2.5f, 4.5f,          // acc
            9999f, 2.0f          // geometry
        )
        // Mock 3 lookahead features (straight road)
        val safeLookahead = LookaheadResult(0f, 9999f, 9999f)
        
        val fusedSafe = sensorFusionEngine.fuse(safeImu, safeLookahead)
        val normalizedSafe = sensorFusionEngine.normalizeAll(fusedSafe)
        val predSafe = modelRunner.predict(normalizedSafe)
        
        Log.d(TAG, "Safe test (12f): ${predSafe.contentToString()}")
        Log.d(TAG, "Predicted class: ${predSafe.indices.maxByOrNull { predSafe[it] }}")
        
        // 2. Mock 9 IMU features (dangerous)
        val dangerImu = floatArrayOf(
            25.0f, 24.0f, 2.0f,  // speed
            5.0f, 10.0f,         // gyro
            5.0f, 8.0f,          // acc
            50.0f, 15.0f         // geometry
        )
        // Mock 3 lookahead features (sharp curve soon)
        val dangerLookahead = LookaheadResult(10f, 45f, 30f)
        
        val fusedDanger = sensorFusionEngine.fuse(dangerImu, dangerLookahead)
        val normalizedDanger = sensorFusionEngine.normalizeAll(fusedDanger)
        val predDanger = modelRunner.predict(normalizedDanger)
        
        Log.d(TAG, "Danger test (12f): ${predDanger.contentToString()}")
        Log.d(TAG, "Predicted class: ${predDanger.indices.maxByOrNull { predDanger[it] }}")
        
        Log.d(TAG, "=== Test Complete ===")
    }
}
