package com.example.driverwarning

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log

/**
 * Manages haptic alerts based on model predictions.
 * 
 * Implements cooldown and confidence thresholds to avoid alert fatigue.
 */
class AlertManager(private val context: Context) {
    
    private val vibrator: Vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        vibratorManager.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
    }
    
    private var lastAlertTime = 0L
    private var lastAlertClass = CLASS_SAFE // Track last alert severity
    private val cooldownMs = 5000L // 5 seconds between alerts
    private val confidenceThreshold = 0.4f // Lowered to 40% for testing
    
    companion object {
        private const val TAG = "AlertManager"
        
        // Severity classes
        const val CLASS_SAFE = 0
        const val CLASS_MILD = 1
        const val CLASS_URGENT = 2
        const val CLASS_HECTIC = 3
    }
    
    // ...
    
    fun processPrediction(context: Context, predictedClass: Int, probability: Float) {
        // Load vehicle settings
        val prefs = context.getSharedPreferences("driver_warning_prefs", Context.MODE_PRIVATE)
        val vehicleType = prefs.getString("vehicle_type", "car") ?: "car"
        
        // Adjust thresholds based on vehicle
        var dynamicThreshold = confidenceThreshold
        
        when (vehicleType) {
            "bike" -> {
                // Bikes are vulnerable: Warn earlier
                dynamicThreshold = 0.35f 
            }
            "truck" -> {
                // Trucks risk tipping: Treat "Urgent" as critical
                if (predictedClass >= CLASS_URGENT) {
                    dynamicThreshold = 0.3f 
                }
            }
            else -> {
                // Car (Standard)
                dynamicThreshold = 0.4f
            }
        }

        // Check if unsafe
        if (predictedClass > CLASS_SAFE && probability > dynamicThreshold) {
            val currentTime = System.currentTimeMillis()
            
            val isEscalating = predictedClass > lastAlertClass
            val isCooldownOver = (currentTime - lastAlertTime) > cooldownMs
            
            if (isEscalating || isCooldownOver) {
                // Trigger appropriate alert
                when (predictedClass) {
                    CLASS_MILD -> {
                        Log.i(TAG, "⚠️ Triggering MILD alert")
                        triggerMildAlert()
                    }
                    CLASS_URGENT -> {
                        Log.w(TAG, "⚠️⚠️ Triggering URGENT alert")
                        triggerUrgentAlert()
                    }
                    CLASS_HECTIC -> {
                        Log.e(TAG, "🚨 Triggering HECTIC alert")
                        triggerHecticAlert()
                    }
                }
                lastAlertTime = currentTime
                lastAlertClass = predictedClass
            }
        } else {
            // Calm down state if safe for a while
            if (System.currentTimeMillis() - lastAlertTime > 2000) {
                lastAlertClass = CLASS_SAFE
            }
        }
    }
    
    /**
     * Reset alert state (call when danger passes or monitoring stops)
     */
    fun resetAlertState() {
        lastAlertClass = CLASS_SAFE
        lastAlertTime = 0L
        Log.d(TAG, "Alert state reset")
    }
    
    /**
     * Mild alert: Three 400ms pulses (more noticeable).
     */
    private fun triggerMildAlert() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val pattern = longArrayOf(0, 400, 200, 400, 200, 400)
            val amplitudes = intArrayOf(0, 180, 0, 180, 0, 180)
            val effect = VibrationEffect.createWaveform(pattern, amplitudes, -1)
            vibrator.vibrate(effect)
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(longArrayOf(0, 400, 200, 400, 200, 400), -1)
        }
    }
    
    /**
     * Urgent alert: Four 500ms strong pulses with short gaps.
     */
    private fun triggerUrgentAlert() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val pattern = longArrayOf(0, 500, 150, 500, 150, 500, 150, 500)
            val amplitudes = intArrayOf(0, 220, 0, 220, 0, 220, 0, 220)
            val effect = VibrationEffect.createWaveform(pattern, amplitudes, -1)
            vibrator.vibrate(effect)
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(longArrayOf(0, 500, 150, 500, 150, 500, 150, 500), -1)
        }
    }
    
    /**
     * Hectic alert: Continuous strong vibration for 2 seconds.
     */
    private fun triggerHecticAlert() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val pattern = longArrayOf(0, 2000)
            val amplitudes = intArrayOf(0, 255) // Maximum amplitude
            val effect = VibrationEffect.createWaveform(pattern, amplitudes, -1)
            vibrator.vibrate(effect)
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(2000)
        }
    }
    
    /**
     * Test vibration pattern (for debug UI).
     */
    fun testVibration(severityClass: Int) {
        when (severityClass) {
            CLASS_MILD -> triggerMildAlert()
            CLASS_URGENT -> triggerUrgentAlert()
            CLASS_HECTIC -> triggerHecticAlert()
        }
    }
}
