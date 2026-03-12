package com.example.driverwarning

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.google.android.gms.location.*
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * Background service for sensor collection and inference.
 */
class SensorService : Service(), SensorEventListener {
    
    private lateinit var sensorManager: SensorManager
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var modelRunner: TFLiteModelRunner
    private lateinit var featureExtractor: FeatureExtractor
    private lateinit var alertManager: AlertManager
    private lateinit var dataLogger: DataLogger

    // ---- Route-aware fusion components ------------------------------------
    private val routeManager = RouteManager()
    private val routeCurvePredictor = RouteCurvePredictor()
    private val sensorFusionEngine = SensorFusionEngine()
    // -----------------------------------------------------------------------
    
    private val sensorBuffer = ConcurrentLinkedQueue<SensorSample>()
    private val bufferDurationMs = 3000L // 3 seconds
    
    private var lastAccelData = FloatArray(3)
    private var lastGyroData = FloatArray(3)
    private var lastLocation: Location? = null
    
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    companion object {
        private const val TAG = "SensorService"
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "driver_warning_channel"
        /** Broadcast action: lookahead state update for the UI warning bar. */
        const val ACTION_LOOKAHEAD_UPDATE = "com.example.driverwarning.LOOKAHEAD_UPDATE"
    }
    
    private var wakeLock: android.os.PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service created")
        
        // Acquire WakeLock
        try {
            val powerManager = getSystemService(android.os.PowerManager::class.java)
            wakeLock = powerManager.newWakeLock(android.os.PowerManager.PARTIAL_WAKE_LOCK, "DriverWarning::SensorServiceWakeLock")
            wakeLock?.acquire(10 * 60 * 1000L /*10 minutes*/)
        } catch (e: Exception) {
            Log.e(TAG, "Error acquiring WakeLock", e)
        }
        
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        modelRunner = TFLiteModelRunner(this)
        featureExtractor = FeatureExtractor()
        alertManager = AlertManager(this)
        dataLogger = DataLogger(this)
        
        // Start logging immediately when service starts
        dataLogger.startLogging()
        
        // Test model to verify it's working
        ModelTest.testModel(modelRunner, featureExtractor)
        
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, createNotification())
        
        startSensorCollection()
        startInferenceLoop()
    }
    
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Driver Warning Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Monitors driving behavior and provides alerts"
            }
            
            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(channel)
        }
    }
    
    private fun createNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Driver Warning Active")
            .setContentText("Monitoring driving behavior...")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }
    
    private fun startSensorCollection() {
        Log.d(TAG, "Starting sensor collection")
        
        // Register accelerometer
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let { accel ->
            sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_GAME)
            Log.d(TAG, "Accelerometer registered")
        } ?: Log.w(TAG, "Accelerometer not available")
        
        // Register gyroscope
        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let { gyro ->
            sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME)
            Log.d(TAG, "Gyroscope registered")
        } ?: Log.w(TAG, "Gyroscope not available")
        
        // Start location updates
        val locationRequest = LocationRequest.Builder(100L) // 10 Hz
            .setPriority(Priority.PRIORITY_HIGH_ACCURACY)
            .build()
        
        try {
            fusedLocationClient.requestLocationUpdates(
                locationRequest,
                locationCallback,
                mainLooper
            )
            Log.d(TAG, "Location updates started")
        } catch (e: SecurityException) {
            Log.e(TAG, "Location permission not granted", e)
        }
    }
    
    private var lastStableLocation: Location? = null

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            result.lastLocation?.let { location ->
                lastLocation = location
                
                // Jitter Filter: If speed < 1 m/s (~3.6 km/h), verify if we really moved
                // to prevent map jumping while stationary (GPS drift).
                if (location.speed >= 1.0f) {
                    lastStableLocation = location
                } else {
                    // We are likely stopped. 
                    // If we have a stable location, use it to pinned the map
                    // effectively ignoring small GPS drifts.
                    // Only update if drift is huge (> 20m) which implies we actually moved slowly
                    if (lastStableLocation != null && location.distanceTo(lastStableLocation!!) < 20f) {
                        // Use stable location but keep the 0 speed from current update
                        // We create a copy to not mess up the original reference
                        val pinnedLocation = Location(location)
                        pinnedLocation.latitude = lastStableLocation!!.latitude
                        pinnedLocation.longitude = lastStableLocation!!.longitude
                        pinnedLocation.speed = 0f // Force 0 speed at stop
                        pinnedLocation.bearing = lastStableLocation!!.bearing // Keep bearing
                        lastLocation = pinnedLocation
                    } else {
                        // First lock or large drift
                        lastStableLocation = location
                    }
                }
                
                val broadcastLoc = lastLocation!!
                addSensorSample()
                
                // Broadcast location update to MainActivity
                val intent = Intent(MainActivity.ACTION_LOCATION_UPDATE).apply {
                    putExtra("lat", broadcastLoc.latitude)
                    putExtra("lon", broadcastLoc.longitude)
                    putExtra("speed", broadcastLoc.speed)
                    putExtra("bearing", broadcastLoc.bearing)
                }
                LocalBroadcastManager.getInstance(this@SensorService).sendBroadcast(intent)
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
    
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not used
    }
    
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
        
        // Remove old samples (keep only last 3 seconds)
        val cutoffTime = System.currentTimeMillis() - bufferDurationMs
        while (sensorBuffer.peek()?.timestamp ?: Long.MAX_VALUE < cutoffTime) {
            sensorBuffer.poll()
        }
        
        Log.v(TAG, "Buffer size: ${sensorBuffer.size}")
    }
    
    private fun startInferenceLoop() {
        Log.d(TAG, "Starting inference loop")
        
        serviceScope.launch {
            while (isActive) {
                delay(1000L) // Run inference every 1 second
                
                val window = sensorBuffer.toList()
                if (window.size >= 5) { // Reduced from 30 to 5 for faster response
                    runInference(window)
                } else {
                    Log.v(TAG, "Insufficient samples: ${window.size}/5 (need location data)")
                }
            }
        }
    }
    
    private var firstPrediction = true

    /**
     * Allow MainActivity to pass the route manager instance after a route is
     * fetched so this service can use the polyline for lookahead scanning.
     * (Alternative: use a shared singleton or a bound service.)
     */
    fun setRoute(polyline: List<com.google.android.gms.maps.model.LatLng>) {
        routeManager.getPolyline()  // ignored; we accept external polylines
        // Store it in the manager by calling internal setter via a workaround:
        // We set a local variable directly since RouteManager is our own class.
        _externalPolyline = polyline
        Log.d(TAG, "Route set externally: ${polyline.size} pts")
    }

    @Volatile private var _externalPolyline: List<com.google.android.gms.maps.model.LatLng> = emptyList()
    
    private fun runInference(window: List<SensorSample>) {
        // Speed Gate: Don't alert if stationary or moving very slowly (< 2 m/s ~ 7 km/h)
        val currentSpeed = window.lastOrNull()?.speed ?: 0f
        if (currentSpeed < 2.0f) {
            Log.v(TAG, "Speed too low for alerts: $currentSpeed m/s")
            return
        }

        try {
            // 1. Extract 9 raw IMU features from sensor window
            val imuFeatures = featureExtractor.extractFeatures(window)
            Log.d(TAG, "IMU features: ${imuFeatures.contentToString()}")

            // 2. Compute route-lookahead features (if a route is loaded)
            val lastLoc = lastLocation
            val polyline = _externalPolyline.takeIf { it.size >= 3 }
                ?: routeManager.getPolyline().takeIf { it.size >= 3 }

            val lookahead: LookaheadResult? = if (polyline != null && lastLoc != null) {
                routeCurvePredictor.scanAhead(
                    polyline,
                    lastLoc.latitude,
                    lastLoc.longitude,
                )
            } else null

            if (lookahead != null) {
                Log.d(TAG, "Lookahead: ${sensorFusionEngine.describeState(lookahead)}")
            } else {
                Log.v(TAG, "No route loaded — IMU-only mode.")
            }

            // 3. Fuse IMU + lookahead → 12-feature vector
            val fused12 = sensorFusionEngine.fuse(imuFeatures, lookahead)
            Log.d(TAG, "Fused 12: ${fused12.contentToString()}")

            // 4. Normalize all 12 features
            val normalized12 = sensorFusionEngine.normalizeAll(fused12)
            Log.d(TAG, "Normalized: ${normalized12.contentToString()}")

            // 5. Run TFLite inference (expects 12 features)
            val probabilities = modelRunner.predict(normalized12)
            Log.d(TAG, "Probabilities: ${probabilities.contentToString()}")

            // 6. Get predicted class
            val predictedClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val confidence = probabilities[predictedClass]
            val classNames = arrayOf("Safe", "Mild", "Urgent", "Hectic")
            Log.i(TAG, "Prediction: ${classNames[predictedClass]} ($confidence)")

            // Show toast on first successful prediction
            if (firstPrediction) {
                firstPrediction = false
                android.os.Handler(android.os.Looper.getMainLooper()).post {
                    android.widget.Toast.makeText(
                        this@SensorService,
                        "🎯 Inference active! ${classNames[predictedClass]}",
                        android.widget.Toast.LENGTH_LONG,
                    ).show()
                }
            }

            // 7. Broadcast prediction to MainActivity
            val predIntent = Intent(MainActivity.ACTION_PREDICTION_UPDATE).apply {
                putExtra("prediction", classNames[predictedClass])
                putExtra("confidence", confidence)
            }
            LocalBroadcastManager.getInstance(this@SensorService).sendBroadcast(predIntent)

            // 8. Broadcast lookahead state to MainActivity (for UI warning bar)
            if (lookahead != null) {
                val laIntent = Intent(ACTION_LOOKAHEAD_UPDATE).apply {
                    putExtra("lookahead_severity", lookahead.lookaheadSeverity)
                    putExtra("map_curve_radius", lookahead.mapCurveRadius)
                    putExtra("distance_to_next_curve", lookahead.distanceToNextCurve)
                }
                LocalBroadcastManager.getInstance(this@SensorService).sendBroadcast(laIntent)
            }

            // 9. Log sample for future retraining dataset
            val lastSample = window.last()
            dataLogger.logSample(lastSample, classNames[predictedClass], confidence)

            // 10. Trigger haptic alert if needed
            alertManager.processPrediction(this@SensorService, predictedClass, confidence)

        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "Service destroyed")
        
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
        }
        
        sensorManager.unregisterListener(this)
        fusedLocationClient.removeLocationUpdates(locationCallback)
        alertManager.resetAlertState() // Reset alert state for next session
        
        dataLogger.stopLogging() // Save and close CSV file
        modelRunner.close()
        serviceScope.cancel()
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
}

