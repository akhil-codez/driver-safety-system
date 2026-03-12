package com.example.driverwarning

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.SupportMapFragment
import com.google.android.gms.maps.model.LatLng
import com.google.android.gms.maps.model.MarkerOptions
import com.google.android.gms.maps.model.Marker

class MainActivity : AppCompatActivity(), OnMapReadyCallback, android.hardware.SensorEventListener {
    
    private var isServiceRunning = false
    private lateinit var statusText: TextView
    private lateinit var locationText: TextView
    private lateinit var predictionText: TextView
    private lateinit var lookaheadText: TextView        // ← new: map-ahead warning bar
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var debugButton: ImageButton
    private lateinit var settingsButton: ImageButton
    
    private var googleMap: GoogleMap? = null
    private var currentMarker: Marker? = null
    private var isFirstLocation = true

    // Route-aware components
    private val routeManager = RouteManager()
    private val routeCurvePredictor = RouteCurvePredictor()
    
    private lateinit var sensorManager: android.hardware.SensorManager
    private var rotationVector: android.hardware.Sensor? = null
    
    companion object {
        private const val LOCATION_PERMISSION_REQUEST_CODE = 1001
        const val ACTION_LOCATION_UPDATE = "com.example.driverwarning.LOCATION_UPDATE"
        const val ACTION_PREDICTION_UPDATE = "com.example.driverwarning.PREDICTION_UPDATE"
        // Google Maps/Directions API key — same key used for the Maps SDK
        private const val MAPS_API_KEY = "AIzaSyCTNy1gFEL2S1gwqrVlUQ1-sw4dEcw5xGA"
    }
    
    private var currentBearing = 0f
    
    override fun onResume() {
        super.onResume()
        rotationVector?.let {
            sensorManager.registerListener(this, it, android.hardware.SensorManager.SENSOR_DELAY_UI)
        }
    }
    
    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }
    
    override fun onSensorChanged(event: android.hardware.SensorEvent?) {
        if (event?.sensor?.type == android.hardware.Sensor.TYPE_ROTATION_VECTOR) {
            val rotationMatrix = FloatArray(9)
            android.hardware.SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
            
            val orientation = FloatArray(3)
            android.hardware.SensorManager.getOrientation(rotationMatrix, orientation)
            
            val azimuthDeg = Math.toDegrees(orientation[0].toDouble()).toFloat()
            val compassBearing = (azimuthDeg + 360) % 360
            
            if (lastKnownSpeed < 1.0f) {
                currentBearing = compassBearing
                googleMap?.let { map ->
                   currentMarker?.rotation = currentBearing
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: android.hardware.Sensor?, accuracy: Int) {
        // Not used
    }

    private var lastKnownSpeed = 0f

    private val updateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                ACTION_LOCATION_UPDATE -> {
                    val lat   = intent.getDoubleExtra("lat", 0.0)
                    val lon   = intent.getDoubleExtra("lon", 0.0)
                    val speed = intent.getFloatExtra("speed", 0f)
                    val bearing = intent.getFloatExtra("bearing", 0f)
                    
                    lastKnownSpeed = speed
                    if (speed > 1.0) currentBearing = bearing
                    
                    locationText.text = "📍 %.6f, %.6f | %.1f m/s (%.0f km/h)".format(
                        lat, lon, speed, speed * 3.6
                    )
                    updateMapLocation(lat, lon)
                }
                ACTION_PREDICTION_UPDATE -> {
                    val prediction = intent.getStringExtra("prediction") ?: "Unknown"
                    val confidence = intent.getFloatExtra("confidence", 0f)
                    predictionText.text = "⚠️ $prediction (${(confidence * 100).toInt()}%)"
                }
                SensorService.ACTION_LOOKAHEAD_UPDATE -> {
                    val dist = intent.getFloatExtra("distance_to_next_curve",
                        RouteCurvePredictor.NO_CURVE_DIST)
                    val radius = intent.getFloatExtra("map_curve_radius",
                        RouteCurvePredictor.STRAIGHT_RADIUS)
                    if (dist >= RouteCurvePredictor.NO_CURVE_DIST - 1f) {
                        lookaheadText.text = "🗺️ Road ahead: clear"
                        lookaheadText.setBackgroundColor(0xFF1A3A2A.toInt())
                    } else {
                        val label = when {
                            radius < 100  -> "🔴 SHARP curve"
                            radius < 200  -> "🟠 Moderate curve"
                            else          -> "🟡 Mild curve"
                        }
                        lookaheadText.text = "$label in ${dist.toInt()} m"
                        val color = when {
                            radius < 100  -> 0xFF5A0000.toInt()
                            radius < 200  -> 0xFF5A3000.toInt()
                            else          -> 0xFF4A4A00.toInt()
                        }
                        lookaheadText.setBackgroundColor(color)
                    }
                }
            }
        }
    }
    
    private fun computeOffset(start: LatLng, distanceMeters: Double, heading: Double): LatLng {
        val earthRadius = 6371000.0
        val latRad = Math.toRadians(start.latitude)
        val lonRad = Math.toRadians(start.longitude)
        val headingRad = Math.toRadians(heading)
        val angDist = distanceMeters / earthRadius

        val newLatRad = Math.asin(Math.sin(latRad) * Math.cos(angDist) +
                Math.cos(latRad) * Math.sin(angDist) * Math.cos(headingRad))
        val newLonRad = lonRad + Math.atan2(Math.sin(headingRad) * Math.sin(angDist) * Math.cos(latRad),
                Math.cos(angDist) - Math.sin(latRad) * Math.sin(newLatRad))

        return LatLng(Math.toDegrees(newLatRad), Math.toDegrees(newLonRad))
    }

    private fun simulateRoute() {
        if (googleMap == null) {
            android.widget.Toast.makeText(this, "⚠️ Map not ready yet!", android.widget.Toast.LENGTH_SHORT).show()
            return
        }
        
        isNavigating = true
        
        googleMap?.let { map ->
            map.clear()
            
            val startLat = currentMarker?.position?.latitude ?: 37.7749
            val startLon = currentMarker?.position?.longitude ?: -122.4194
            val startPos = LatLng(startLat, startLon)
            
            val p1 = computeOffset(startPos, 100.0, currentBearing.toDouble())
            val p2 = computeOffset(p1, 50.0, (currentBearing + 45).toDouble())
            val p3 = computeOffset(p2, 100.0, (currentBearing + 45).toDouble())
            
            val polylineOptions = com.google.android.gms.maps.model.PolylineOptions()
                .add(startPos).add(p1).add(p2).add(p3)
                .color(android.graphics.Color.BLUE).width(15f)
            map.addPolyline(polylineOptions)
            
            map.addMarker(MarkerOptions().position(startPos).title("Start")
                .icon(com.google.android.gms.maps.model.BitmapDescriptorFactory
                    .defaultMarker(com.google.android.gms.maps.model.BitmapDescriptorFactory.HUE_GREEN)))
            map.addMarker(MarkerOptions().position(p3).title("Destination")
                .icon(com.google.android.gms.maps.model.BitmapDescriptorFactory
                    .defaultMarker(com.google.android.gms.maps.model.BitmapDescriptorFactory.HUE_RED)))
            map.addMarker(MarkerOptions().position(p1).title("⚠️ Sharp Curve Detected")
                .icon(com.google.android.gms.maps.model.BitmapDescriptorFactory
                    .defaultMarker(com.google.android.gms.maps.model.BitmapDescriptorFactory.HUE_ORANGE)))
            
            val bounds = com.google.android.gms.maps.model.LatLngBounds.Builder()
                .include(startPos).include(p1).include(p3).build()
            try {
                map.animateCamera(CameraUpdateFactory.newLatLngBounds(bounds, 200))
            } catch (e: Exception) {
                map.moveCamera(CameraUpdateFactory.newLatLngZoom(startPos, 17f))
            }
            android.widget.Toast.makeText(this, "⚠️ DEMO: Simulated Sharp Curve Ahead generated!", android.widget.Toast.LENGTH_LONG).show()
        }
    }

    /**
     * Fetch a real route from the Google Directions API, draw it on the map,
     * pre-scan for sharp curves and mark them, then hand the polyline to
     * SensorService for live lookahead inference.
     */
    private fun fetchAndDisplayRoute(destination: String) {
        val originLat = currentMarker?.position?.latitude ?: 37.7749
        val originLon = currentMarker?.position?.longitude ?: -122.4194
        val origin = "$originLat,$originLon"

        routeManager.fetchRoute(
            origin = origin,
            destination = destination,
            apiKey = MAPS_API_KEY,
            scope = lifecycleScope,
        ) { success ->
            if (!success) {
                statusText.text = "🗺️ API failed — using demo route"
                simulateRoute()
                return@fetchRoute
            }

            val polyline = routeManager.getPolyline()
            val radii = routeCurvePredictor.computeRadii(polyline)
            
            isNavigating = true
            statusText.text = "🗺️ NAVIGATING (${polyline.size} pts)"

            googleMap?.let { map ->
                map.clear()

                // Draw blue route polyline
                val opts = com.google.android.gms.maps.model.PolylineOptions()
                    .addAll(polyline)
                    .color(android.graphics.Color.BLUE)
                    .width(10f)
                map.addPolyline(opts)

                // Start / destination markers
                if (polyline.isNotEmpty()) {
                    map.addMarker(MarkerOptions().position(polyline.first()).title("Start")
                        .icon(com.google.android.gms.maps.model.BitmapDescriptorFactory
                            .defaultMarker(com.google.android.gms.maps.model.BitmapDescriptorFactory.HUE_GREEN)))
                    map.addMarker(MarkerOptions().position(polyline.last()).title("Destination")
                        .icon(com.google.android.gms.maps.model.BitmapDescriptorFactory
                            .defaultMarker(com.google.android.gms.maps.model.BitmapDescriptorFactory.HUE_RED)))
                }

                // Pre-scan curve segments and mark sharp ones with orange markers
                var lastMarkedIdx = -50
                radii.forEachIndexed { i, radius ->
                    if (radius < RouteCurvePredictor.CURVE_THRESHOLD_M && i - lastMarkedIdx > 10) {
                        map.addMarker(
                            MarkerOptions()
                                .position(polyline[i])
                                .title("⚠️ Curve (r=${radius.toInt()} m)")
                                .icon(
                                    com.google.android.gms.maps.model.BitmapDescriptorFactory
                                        .defaultMarker(
                                            if (radius < 100f)
                                                com.google.android.gms.maps.model.BitmapDescriptorFactory.HUE_RED
                                            else
                                                com.google.android.gms.maps.model.BitmapDescriptorFactory.HUE_ORANGE
                                        )
                                )
                        )
                        lastMarkedIdx = i
                    }
                }

                // Fit camera to full route
                if (polyline.size >= 2) {
                    val boundsBuilder = com.google.android.gms.maps.model.LatLngBounds.Builder()
                    polyline.forEach { boundsBuilder.include(it) }
                    try {
                        map.animateCamera(
                            CameraUpdateFactory.newLatLngBounds(boundsBuilder.build(), 80)
                        )
                    } catch (e: Exception) {
                        map.moveCamera(CameraUpdateFactory.newLatLngZoom(polyline.first(), 14f))
                    }
                }
            }

            // Hand the route polyline to SensorService for live lookahead inference
            // SensorService holds its own RouteManager internally; we pass via broadcast
            val routeIntent = Intent("com.example.driverwarning.SET_ROUTE").apply {
                // Encode as parallel float arrays for the broadcast
                putExtra("lats", FloatArray(polyline.size) { i -> polyline[i].latitude.toFloat() })
                putExtra("lons", FloatArray(polyline.size) { i -> polyline[i].longitude.toFloat() })
            }
            androidx.localbroadcastmanager.content.LocalBroadcastManager
                .getInstance(this).sendBroadcast(routeIntent)

            val curveCount = radii.count { it < RouteCurvePredictor.CURVE_THRESHOLD_M }
            android.widget.Toast.makeText(
                this,
                "🗺️ Route: ${polyline.size} pts, $curveCount curves pre-detected",
                android.widget.Toast.LENGTH_LONG,
            ).show()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        enableFullscreen()
        setContentView(R.layout.activity_main)
        
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as android.hardware.SensorManager
        rotationVector = sensorManager.getDefaultSensor(android.hardware.Sensor.TYPE_ROTATION_VECTOR)
        
        initViews()
        initMap()
        checkPermissions()
        setupListeners()
        registerReceivers()
    }
    
    private fun enableFullscreen() {
        supportActionBar?.hide()
        
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            window.setDecorFitsSystemWindows(false)
            window.insetsController?.let {
                it.hide(android.view.WindowInsets.Type.statusBars() or android.view.WindowInsets.Type.navigationBars())
                it.systemBarsBehavior = android.view.WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                android.view.View.SYSTEM_UI_FLAG_FULLSCREEN
                or android.view.View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or android.view.View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                or android.view.View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                or android.view.View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                or android.view.View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
            )
        }
    }
    
    private lateinit var navButton: ImageButton
    private lateinit var layersButton: ImageButton
    private lateinit var destinationContainer: android.view.View
    private lateinit var destinationInput: android.widget.EditText
    private lateinit var startNavButton: Button
    private lateinit var cancelNavButton: Button
    
    private fun initViews() {
        statusText    = findViewById(R.id.statusText)
        locationText  = findViewById(R.id.locationText)
        predictionText = findViewById(R.id.predictionText)
        // lookaheadText may not exist in older layout versions; create dynamically if needed
        lookaheadText = try {
            findViewById(R.id.lookaheadText)
        } catch (e: Exception) {
            // Fallback: reuse predictionText as a combined field
            predictionText
        }
        startButton   = findViewById(R.id.startButton)
        stopButton    = findViewById(R.id.stopButton)
        debugButton   = findViewById(R.id.debugButton)
        settingsButton = findViewById(R.id.settingsButton)
        
        navButton           = findViewById(R.id.navButton)
        layersButton        = findViewById(R.id.layersButton)
        destinationContainer = findViewById(R.id.destinationContainer)
        destinationInput    = findViewById(R.id.destinationInput)
        startNavButton      = findViewById(R.id.startNavButton)
        cancelNavButton     = findViewById(R.id.cancelNavButton)
        
        updateUI()
    }
    
    private fun initMap() {
        val mapFragment = supportFragmentManager
            .findFragmentById(R.id.mapFragment) as? SupportMapFragment
        mapFragment?.getMapAsync(this)
    }
    
    override fun onMapReady(map: GoogleMap) {
        googleMap = map
        
        map.uiSettings.isZoomControlsEnabled = false
        map.uiSettings.isCompassEnabled = true
        map.uiSettings.isMyLocationButtonEnabled = false
        
        try {
            map.isTrafficEnabled = true
            map.isBuildingsEnabled = true
        } catch (e: SecurityException) {
            // Permission might be missing
        }
        
        val defaultLocation = LatLng(37.7749, -122.4194)
        map.moveCamera(CameraUpdateFactory.newLatLngZoom(defaultLocation, 15f))
    }
    
    private var isNavigating = false

    private fun updateMapLocation(lat: Double, lon: Double) {
        googleMap?.let { map ->
            val location = LatLng(lat, lon)
            
            currentMarker?.remove()
            
            currentMarker = map.addMarker(
                MarkerOptions()
                    .position(location)
                    .title("Current Location")
                    .rotation(currentBearing)
                    .flat(true)
                    .anchor(0.5f, 0.5f)
            )
            
            if (isNavigating) {
                val cameraPosition = com.google.android.gms.maps.model.CameraPosition.Builder()
                    .target(location)
                    .zoom(18f)
                    .bearing(currentBearing)
                    .tilt(60f)
                    .build()
                
                map.animateCamera(CameraUpdateFactory.newCameraPosition(cameraPosition), 1000, null)
                
            } else {
                if (isFirstLocation) {
                    map.animateCamera(CameraUpdateFactory.newLatLngZoom(location, 17f))
                    isFirstLocation = false
                } else {
                    map.animateCamera(CameraUpdateFactory.newLatLng(location))
                }
            }
        }
    }
    
    private fun registerReceivers() {
        val filter = IntentFilter().apply {
            addAction(ACTION_LOCATION_UPDATE)
            addAction(ACTION_PREDICTION_UPDATE)
            addAction(SensorService.ACTION_LOOKAHEAD_UPDATE)  // ← route-lookahead updates
        }
        LocalBroadcastManager.getInstance(this).registerReceiver(updateReceiver, filter)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        LocalBroadcastManager.getInstance(this).unregisterReceiver(updateReceiver)
    }
    
    private fun setupListeners() {
        startButton.setOnClickListener {
            if (hasLocationPermission()) {
                startSensorService()
            } else {
                requestLocationPermission()
            }
        }
        
        stopButton.setOnClickListener {
            stopSensorService()
        }
        
        debugButton.setOnClickListener {
            startActivity(Intent(this, DebugActivity::class.java))
        }
        
        settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        
        navButton.setOnClickListener {
            destinationContainer.visibility = android.view.View.VISIBLE
        }
        
        layersButton.setOnClickListener {
            val options = arrayOf("Normal", "Satellite", "Hybrid", "Terrain")
            android.app.AlertDialog.Builder(this)
                .setTitle("Select Map Type")
                .setItems(options) { _, which ->
                    googleMap?.mapType = when (which) {
                        0 -> GoogleMap.MAP_TYPE_NORMAL
                        1 -> GoogleMap.MAP_TYPE_SATELLITE
                        2 -> GoogleMap.MAP_TYPE_HYBRID
                        3 -> GoogleMap.MAP_TYPE_TERRAIN
                        else -> GoogleMap.MAP_TYPE_NORMAL
                    }
                }
                .show()
        }
        
        cancelNavButton.setOnClickListener {
            destinationContainer.visibility = android.view.View.GONE
            isNavigating = false
            currentMarker?.position?.let { 
                 googleMap?.animateCamera(CameraUpdateFactory.newLatLngZoom(it, 17f))
            }
        }
        
        startNavButton.setOnClickListener {
            val input = destinationInput.text.toString()
            if (input.isNotEmpty()) {
                destinationContainer.visibility = android.view.View.GONE
                statusText.text = "🗺️ FETCHING ROUTE..."
                fetchAndDisplayRoute(input)
            }
        }
    }
    
    private fun checkPermissions() {
        if (!hasLocationPermission()) {
            requestLocationPermission()
        }
    }
    
    private fun hasLocationPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    private fun requestLocationPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ),
            LOCATION_PERMISSION_REQUEST_CODE
        )
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        if (requestCode == LOCATION_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                statusText.text = "🚗 Driver Warning System"
            } else {
                statusText.text = "⚠️ Location permission required"
            }
        }
    }
    
    private fun startSensorService() {
        val intent = Intent(this, SensorService::class.java)
        ContextCompat.startForegroundService(this, intent)
        isServiceRunning = true
        isFirstLocation = true
        updateUI()
    }
    
    private fun stopSensorService() {
        val intent = Intent(this, SensorService::class.java)
        stopService(intent)
        isServiceRunning = false
        locationText.text = "Location: Not tracking"
        predictionText.text = "Prediction: --"
        currentMarker?.remove()
        updateUI()
    }
    
    private fun updateUI() {
        if (isServiceRunning) {
            statusText.text = "🚗 MONITORING ACTIVE"
            locationText.text = "📍 Waiting for GPS signal..."
            
            startButton.visibility = android.view.View.GONE
            stopButton.visibility = android.view.View.VISIBLE
        } else {
            statusText.text = "🚗 Driver Warning System"
            locationText.text = "Press START to begin monitoring"
            
            startButton.visibility = android.view.View.VISIBLE
            stopButton.visibility = android.view.View.GONE
        }
    }
}
