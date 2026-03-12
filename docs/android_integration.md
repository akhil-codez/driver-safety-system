# Android Integration Guide

This guide explains how to integrate the Driver Warning System TFLite model into an Android app.

## Overview

The Android app performs these steps:
1. Collect sensor data (GPS + IMU) at 10 Hz
2. Maintain a 3-second ring buffer
3. Extract features every 1 second
4. Normalize features using scaler parameters
5. Run TFLite inference
6. Trigger haptic alerts based on predictions

## Prerequisites

- Android Studio Arctic Fox or later
- Minimum SDK: 24 (Android 7.0)
- Target SDK: 33 (Android 13)
- Kotlin 1.8+

## Dependencies

Add to `app/build.gradle`:

```gradle
dependencies {
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.12.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.3'
    
    // Coroutines for background processing
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.1'
    
    // Location services
    implementation 'com.google.android.gms:play-services-location:21.0.1'
}
```

## Permissions

Add to `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
<uses-permission android:name="android.permission.VIBRATE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
```

## Step 1: Add TFLite Model to Assets

1. Copy `models/curve_detector.tflite` to `app/src/main/assets/`
2. The model will be packaged with your APK

## Step 2: Load Scaler Parameters

Extract scaler parameters from Python:

```python
import pickle
import json

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist()
}

with open('models/scaler_params.json', 'w') as f:
    json.dump(scaler_params, f)
```

Add `scaler_params.json` to assets or hardcode in Kotlin:

```kotlin
object ScalerParams {
    val mean = floatArrayOf(
        // Extract from scaler.mean_
        15.0f,  // current_speed
        14.5f,  // mean_speed
        2.0f,   // speed_std
        5.0f,   // gyro_z_mean
        15.0f,  // gyro_z_max
        1.5f,   // acc_y_mean
        3.0f,   // acc_y_max
        500.0f, // curve_radius
        0.5f    // severity_proxy
    )
    
    val scale = floatArrayOf(
        // Extract from scaler.scale_
        5.0f,   // current_speed
        4.5f,   // mean_speed
        1.5f,   // speed_std
        8.0f,   // gyro_z_mean
        12.0f,  // gyro_z_max
        1.2f,   // acc_y_mean
        2.0f,   // acc_y_max
        300.0f, // curve_radius
        0.8f    // severity_proxy
    )
}
```

## Step 3: TFLite Model Runner

Create `TFLiteModelRunner.kt`:

```kotlin
import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteModelRunner(context: Context) {
    private val interpreter: Interpreter
    
    init {
        val model = loadModelFile(context, "curve_detector.tflite")
        interpreter = Interpreter(model)
    }
    
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun predict(features: FloatArray): FloatArray {
        require(features.size == 9) { "Expected 9 features, got ${features.size}" }
        
        // Prepare input: shape [1, 9]
        val inputArray = Array(1) { features }
        
        // Prepare output: shape [1, 4]
        val outputArray = Array(1) { FloatArray(4) }
        
        // Run inference
        interpreter.run(inputArray, outputArray)
        
        return outputArray[0]
    }
    
    fun close() {
        interpreter.close()
    }
}
```

## Step 4: Feature Extraction

Create `FeatureExtractor.kt`:

```kotlin
import kotlin.math.*

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

class FeatureExtractor {
    
    fun extractFeatures(window: List<SensorSample>): FloatArray {
        require(window.isNotEmpty()) { "Window cannot be empty" }
        
        val features = FloatArray(9)
        
        // 1. current_speed
        features[0] = window.last().speed
        
        // 2. mean_speed
        features[1] = window.map { it.speed }.average().toFloat()
        
        // 3. speed_std
        val meanSpeed = features[1]
        val variance = window.map { (it.speed - meanSpeed).pow(2) }.average()
        features[2] = sqrt(variance).toFloat()
        
        // 4. gyro_z_mean
        features[3] = window.map { it.gyroZ }.average().toFloat()
        
        // 5. gyro_z_max
        features[4] = window.map { abs(it.gyroZ) }.maxOrNull() ?: 0f
        
        // 6. acc_y_mean
        features[5] = window.map { it.accY }.average().toFloat()
        
        // 7. acc_y_max
        features[6] = window.map { abs(it.accY) }.maxOrNull() ?: 0f
        
        // 8. curve_radius
        val gpsPoints = window.takeLast(5).map { Pair(it.lat, it.lon) }
        features[7] = computeCurveRadius(gpsPoints)
        
        // 9. severity_proxy
        val radius = maxOf(features[7], 1f)
        features[8] = features[0].pow(2) / radius
        
        return features
    }
    
    private fun computeCurveRadius(points: List<Pair<Double, Double>>): Float {
        if (points.size < 3) return 999999f
        
        // Use last 3 points for Menger curvature
        val p1 = points[points.size - 3]
        val p2 = points[points.size - 2]
        val p3 = points[points.size - 1]
        
        // Compute side lengths
        val a = haversineDistance(p1, p2)
        val b = haversineDistance(p2, p3)
        val c = haversineDistance(p3, p1)
        
        if (a < 0.1 || b < 0.1 || c < 0.1) return 999999f
        
        // Heron's formula for area
        val s = (a + b + c) / 2
        val areaSquared = s * (s - a) * (s - b) * (s - c)
        
        if (areaSquared <= 0) return 999999f
        
        val area = sqrt(areaSquared)
        
        // Menger curvature
        val curvature = (4 * area) / (a * b * c)
        
        if (curvature < 1e-6) return 999999f
        
        val radius = 1.0 / curvature
        return radius.coerceIn(1.0, 999999.0).toFloat()
    }
    
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
    
    fun normalize(features: FloatArray): FloatArray {
        require(features.size == 9) { "Expected 9 features" }
        
        return FloatArray(9) { i ->
            (features[i] - ScalerParams.mean[i]) / ScalerParams.scale[i]
        }
    }
}
```

## Step 5: Sensor Service

Create `SensorService.kt`:

```kotlin
import android.app.Service
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.os.IBinder
import com.google.android.gms.location.*
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentLinkedQueue

class SensorService : Service(), SensorEventListener {
    
    private lateinit var sensorManager: SensorManager
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var modelRunner: TFLiteModelRunner
    private lateinit var featureExtractor: FeatureExtractor
    private lateinit var alertManager: AlertManager
    
    private val sensorBuffer = ConcurrentLinkedQueue<SensorSample>()
    private val bufferDurationMs = 3000L // 3 seconds
    
    private var lastAccelData = FloatArray(3)
    private var lastGyroData = FloatArray(3)
    private var lastLocation: Location? = null
    
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    override fun onCreate() {
        super.onCreate()
        
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        modelRunner = TFLiteModelRunner(this)
        featureExtractor = FeatureExtractor()
        alertManager = AlertManager(this)
        
        startSensorCollection()
        startInferenceLoop()
    }
    
    private fun startSensorCollection() {
        // Register accelerometer
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let { accel ->
            sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_GAME)
        }
        
        // Register gyroscope
        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let { gyro ->
            sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME)
        }
        
        // Start location updates
        val locationRequest = LocationRequest.Builder(100L) // 10 Hz
            .setPriority(Priority.PRIORITY_HIGH_ACCURACY)
            .build()
        
        fusedLocationClient.requestLocationUpdates(
            locationRequest,
            locationCallback,
            mainLooper
        )
    }
    
    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            result.lastLocation?.let { location ->
                lastLocation = location
                addSensorSample()
            }
        }
    }
    
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                lastAccelData = event.values.clone()
            }
            Sensor.TYPE_GYROSCOPE -> {
                lastGyroData = event.values.clone()
            }
        }
    }
    
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    
    private fun addSensorSample() {
        val loc = lastLocation ?: return
        
        val sample = SensorSample(
            timestamp = System.currentTimeMillis(),
            lat = loc.latitude,
            lon = loc.longitude,
            speed = loc.speed,
            heading = loc.bearing,
            accX = lastAccelData[0],
            accY = lastAccelData[1],
            accZ = lastAccelData[2],
            gyroX = Math.toDegrees(lastGyroData[0].toDouble()).toFloat(),
            gyroY = Math.toDegrees(lastGyroData[1].toDouble()).toFloat(),
            gyroZ = Math.toDegrees(lastGyroData[2].toDouble()).toFloat()
        )
        
        sensorBuffer.add(sample)
        
        // Remove old samples
        val cutoffTime = System.currentTimeMillis() - bufferDurationMs
        while (sensorBuffer.peek()?.timestamp ?: Long.MAX_VALUE < cutoffTime) {
            sensorBuffer.poll()
        }
    }
    
    private fun startInferenceLoop() {
        serviceScope.launch {
            while (isActive) {
                delay(1000L) // Run inference every 1 second
                
                val window = sensorBuffer.toList()
                if (window.size >= 30) { // 3 seconds at 10 Hz
                    runInference(window)
                }
            }
        }
    }
    
    private fun runInference(window: List<SensorSample>) {
        try {
            // Extract features
            val features = featureExtractor.extractFeatures(window)
            
            // Normalize
            val normalizedFeatures = featureExtractor.normalize(features)
            
            // Run model
            val probabilities = modelRunner.predict(normalizedFeatures)
            
            // Get predicted class
            val predictedClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            
            // Trigger alert if needed
            alertManager.handlePrediction(predictedClass, probabilities[predictedClass])
            
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        fusedLocationClient.removeLocationUpdates(locationCallback)
        modelRunner.close()
        serviceScope.cancel()
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
}
```

## Step 6: Alert Manager

Create `AlertManager.kt`:

```kotlin
import android.content.Context
import android.os.VibrationEffect
import android.os.Vibrator

class AlertManager(private val context: Context) {
    
    private val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
    private var lastAlertTime = 0L
    private val cooldownMs = 5000L // 5 seconds between alerts
    
    fun handlePrediction(predictedClass: Int, confidence: Float) {
        if (confidence < 0.6f) return // Require 60% confidence
        
        val now = System.currentTimeMillis()
        if (now - lastAlertTime < cooldownMs) return
        
        when (predictedClass) {
            1 -> triggerMildAlert()
            2 -> triggerUrgentAlert()
            3 -> triggerHecticAlert()
        }
        
        lastAlertTime = now
    }
    
    private fun triggerMildAlert() {
        val pattern = longArrayOf(0, 200) // Single 200ms pulse
        vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
    }
    
    private fun triggerUrgentAlert() {
        val pattern = longArrayOf(0, 300, 100, 300) // Two 300ms pulses
        vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
    }
    
    private fun triggerHecticAlert() {
        val pattern = longArrayOf(0, 500) // Strong 500ms pulse
        vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
    }
}
```

## Testing with Emulator

For testing without a real device, create a debug interface:

```kotlin
// In DebugActivity.kt
fun testInference() {
    val testFeatures = floatArrayOf(
        15.0f,  // current_speed
        14.5f,  // mean_speed
        1.2f,   // speed_std
        10.0f,  // gyro_z_mean
        15.0f,  // gyro_z_max
        2.0f,   // acc_y_mean
        3.5f,   // acc_y_max
        150.0f, // curve_radius (sharp curve!)
        1.5f    // severity_proxy
    )
    
    val extractor = FeatureExtractor()
    val normalized = extractor.normalize(testFeatures)
    
    val runner = TFLiteModelRunner(this)
    val probs = runner.predict(normalized)
    
    println("Probabilities: ${probs.contentToString()}")
    // Expected: high probability for "urgent" or "hectic"
}
```

## Summary

✅ Add TFLite model to assets  
✅ Extract and hardcode scaler parameters  
✅ Implement TFLiteModelRunner for inference  
✅ Implement FeatureExtractor with exact Python logic  
✅ Create SensorService for background collection  
✅ Implement AlertManager for haptic feedback  
✅ Test with debug interface  

**Critical**: Ensure feature computation matches Python exactly!
