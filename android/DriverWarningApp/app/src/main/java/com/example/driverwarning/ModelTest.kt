package com.example.driverwarning

import android.util.Log

/**
 * Simple test to verify model and scaler are working
 */
object ModelTest {
    private const val TAG = "ModelTest"
    
    fun testModel(modelRunner: TFLiteModelRunner, featureExtractor: FeatureExtractor) {
        Log.d(TAG, "=== Testing Model ===")
        
        // Test case 1: Safe driving (low speed, straight road)
        val safeFeatures = floatArrayOf(
            10.0f,    // current_speed
            10.0f,    // mean_speed
            0.5f,     // speed_std
            0.1f,     // gyro_z_mean
            0.2f,     // gyro_z_max
            0.5f,     // acc_y_mean
            1.0f,     // acc_y_max
            9999.0f,  // curve_radius (straight)
            0.01f     // severity_proxy
        )
        
        val safeNormalized = featureExtractor.normalize(safeFeatures)
        val safePred = modelRunner.predict(safeNormalized)
        Log.d(TAG, "Safe test: ${safePred.contentToString()}")
        Log.d(TAG, "Predicted class: ${safePred.indices.maxByOrNull { safePred[it] }}")
        
        // Test case 2: Dangerous driving (high speed, sharp curve)
        val dangerousFeatures = floatArrayOf(
            25.0f,    // current_speed (high)
            24.0f,    // mean_speed
            2.0f,     // speed_std
            5.0f,     // gyro_z_mean (turning)
            10.0f,    // gyro_z_max
            5.0f,     // acc_y_mean (lateral)
            8.0f,     // acc_y_max
            50.0f,    // curve_radius (sharp curve)
            12.5f     // severity_proxy (high)
        )
        
        val dangerNormalized = featureExtractor.normalize(dangerousFeatures)
        val dangerPred = modelRunner.predict(dangerNormalized)
        Log.d(TAG, "Dangerous test: ${dangerPred.contentToString()}")
        Log.d(TAG, "Predicted class: ${dangerPred.indices.maxByOrNull { dangerPred[it] }}")
        
        Log.d(TAG, "=== Test Complete ===")
    }
}
