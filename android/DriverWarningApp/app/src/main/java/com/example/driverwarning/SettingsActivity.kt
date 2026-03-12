package com.example.driverwarning

import android.os.Bundle
import android.widget.Switch
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Settings activity for configuring app behavior.
 */
class SettingsActivity : AppCompatActivity() {
    
    private lateinit var hapticSwitch: Switch
    private lateinit var uploadConsentSwitch: Switch
    private lateinit var vehicleTypeGroup: android.widget.RadioGroup
    private lateinit var infoText: TextView
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Enable fullscreen mode
        enableFullscreen()
        
        setContentView(R.layout.activity_settings)
        
        initViews()
        loadSettings()
        setupListeners()
    }
    
    private fun enableFullscreen() {
        supportActionBar?.hide()
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            window.setDecorFitsSystemWindows(false)
            window.insetsController?.hide(android.view.WindowInsets.Type.statusBars() or android.view.WindowInsets.Type.navigationBars())
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                android.view.View.SYSTEM_UI_FLAG_FULLSCREEN
                or android.view.View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or android.view.View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
            )
        }
    }
    
    private fun initViews() {
        hapticSwitch = findViewById(R.id.hapticSwitch)
        uploadConsentSwitch = findViewById(R.id.uploadConsentSwitch)
        vehicleTypeGroup = findViewById(R.id.vehicleTypeGroup) // New
        infoText = findViewById(R.id.infoText)
        
        infoText.text = """
            ⚙️ Settings
            
            ✅ REAL-TIME FEATURES (ACTIVE):
            • Live GPS tracking
            • Real-time curve detection
            • Haptic vibration alerts
            • ML model inference
            
            Vehicle Type: Adjusts alert sensitivity based on vehicle safety profile.
            
            Haptic Feedback: Enable/disable vibration alerts
            
            Data Upload Consent: (Cloud feature - not implemented)
        """.trimIndent()
    }
    
    private fun loadSettings() {
        val prefs = getSharedPreferences("driver_warning_prefs", MODE_PRIVATE)
        hapticSwitch.isChecked = prefs.getBoolean("haptic_enabled", true)
        uploadConsentSwitch.isChecked = prefs.getBoolean("upload_consent", false)
        
        // Load Vehicle Type
        when (prefs.getString("vehicle_type", "car")) {
            "bike" -> vehicleTypeGroup.check(R.id.radioBike)
            "truck" -> vehicleTypeGroup.check(R.id.radioTruck)
            else -> vehicleTypeGroup.check(R.id.radioCar)
        }
    }
    
    private fun setupListeners() {
        hapticSwitch.setOnCheckedChangeListener { _, isChecked ->
            val prefs = getSharedPreferences("driver_warning_prefs", MODE_PRIVATE)
            prefs.edit().putBoolean("haptic_enabled", isChecked).apply()
        }
        
        uploadConsentSwitch.setOnCheckedChangeListener { _, isChecked ->
            val prefs = getSharedPreferences("driver_warning_prefs", MODE_PRIVATE)
            prefs.edit().putBoolean("upload_consent", isChecked).apply()
        }
        
        vehicleTypeGroup.setOnCheckedChangeListener { _, checkedId ->
            val prefs = getSharedPreferences("driver_warning_prefs", MODE_PRIVATE)
            val editor = prefs.edit()
            
            when (checkedId) {
                R.id.radioCar -> editor.putString("vehicle_type", "car")
                R.id.radioBike -> editor.putString("vehicle_type", "bike")
                R.id.radioTruck -> editor.putString("vehicle_type", "truck")
            }
            editor.apply()
        }
    }
}
