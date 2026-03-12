package com.example.driverwarning

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject

/**
 * Debug activity for manual testing of model inference.
 * 
 * Allows entering feature values as JSON and testing vibration patterns.
 */
class DebugActivity : AppCompatActivity() {
    
    private lateinit var featureInput: EditText
    private lateinit var runInferenceButton: Button
    private lateinit var resultText: TextView
    private lateinit var testMildButton: Button
    private lateinit var testUrgentButton: Button
    private lateinit var testHecticButton: Button
    
    private lateinit var modelRunner: TFLiteModelRunner
    private lateinit var featureExtractor: FeatureExtractor
    private lateinit var alertManager: AlertManager
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Enable fullscreen mode
        enableFullscreen()
        
        setContentView(R.layout.activity_debug)
        
        try {
            modelRunner = TFLiteModelRunner(this)
            featureExtractor = FeatureExtractor()
            alertManager = AlertManager(this)
            
            android.widget.Toast.makeText(this, "✅ Model loaded successfully!", android.widget.Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            android.widget.Toast.makeText(this, "❌ Error loading model: ${e.message}", android.widget.Toast.LENGTH_LONG).show()
            android.util.Log.e("DebugActivity", "Model load error", e)
        }
        
        initViews()
        setupListeners()
        setExampleInput()
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
    
    private lateinit var logPathText: TextView
    private lateinit var exportLogButton: Button
    
    private fun initViews() {
        featureInput = findViewById(R.id.featureInput)
        runInferenceButton = findViewById(R.id.runInferenceButton)
        resultText = findViewById(R.id.resultText)
        testMildButton = findViewById(R.id.testMildButton)
        testUrgentButton = findViewById(R.id.testUrgentButton)
        testHecticButton = findViewById(R.id.testHecticButton)
        
        logPathText = findViewById(R.id.logPathText)
        exportLogButton = findViewById(R.id.exportLogButton)
        
        updateLogStatus()
    }
    
    private fun updateLogStatus() {
        val lastFile = DataLogger(this).getLastLogFile()
        if (lastFile != null) {
            logPathText.text = "Found: ${lastFile.name}\nSize: ${lastFile.length() / 1024} KB"
        } else {
            logPathText.text = "No logs found yet"
        }
    }
    
    private fun setupListeners() {
        runInferenceButton.setOnClickListener {
            runManualInference()
        }
        
        exportLogButton.setOnClickListener {
            val lastFile = DataLogger(this).getLastLogFile()
            if (lastFile != null) {
                shareLogFile(lastFile)
            } else {
                android.widget.Toast.makeText(this, "No log file to export", android.widget.Toast.LENGTH_SHORT).show()
            }
        }
        
        testMildButton.setOnClickListener {
            alertManager.testVibration(AlertManager.CLASS_MILD)
            resultText.text = "Triggered MILD vibration"
        }
        
        testUrgentButton.setOnClickListener {
            alertManager.testVibration(AlertManager.CLASS_URGENT)
            resultText.text = "Triggered URGENT vibration"
        }
        
        testHecticButton.setOnClickListener {
            alertManager.testVibration(AlertManager.CLASS_HECTIC)
            resultText.text = "Triggered HECTIC vibration"
        }
    }
    
    private fun shareLogFile(file: java.io.File) {
        try {
            val uri = androidx.core.content.FileProvider.getUriForFile(
                this,
                "${applicationContext.packageName}.fileprovider",
                file
            )
            
            val intent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                type = "text/csv"
                putExtra(android.content.Intent.EXTRA_STREAM, uri)
                addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            
            startActivity(android.content.Intent.createChooser(intent, "Export Training Data"))
            
        } catch (e: Exception) {
            android.util.Log.e("DebugActivity", "Error sharing file", e)
            android.widget.Toast.makeText(this, "Export failed: ${e.message}", android.widget.Toast.LENGTH_LONG).show()
        }
    }
    
    private fun setExampleInput() {
        val example = """
            {
              "current_speed": 15.0,
              "mean_speed": 14.5,
              "speed_std": 1.2,
              "gyro_z_mean": 10.0,
              "gyro_z_max": 15.0,
              "acc_y_mean": 2.0,
              "acc_y_max": 3.5,
              "curve_radius": 150.0,
              "severity_proxy": 1.5
            }
        """.trimIndent()
        
        featureInput.setText(example)
    }
    
    private fun runManualInference() {
        try {
            val json = JSONObject(featureInput.text.toString())
            
            // Extract features from JSON
            val features = floatArrayOf(
                json.getDouble("current_speed").toFloat(),
                json.getDouble("mean_speed").toFloat(),
                json.getDouble("speed_std").toFloat(),
                json.getDouble("gyro_z_mean").toFloat(),
                json.getDouble("gyro_z_max").toFloat(),
                json.getDouble("acc_y_mean").toFloat(),
                json.getDouble("acc_y_max").toFloat(),
                json.getDouble("curve_radius").toFloat(),
                json.getDouble("severity_proxy").toFloat()
            )
            
            // Normalize
            val normalized = featureExtractor.normalize(features)
            
            // Run inference
            val probabilities = modelRunner.predict(normalized)
            
            // Get predicted class
            val predictedClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val classNames = arrayOf("Safe", "Mild", "Urgent", "Hectic")
            
            // Display results
            val result = buildString {
                appendLine("=== Inference Results ===")
                appendLine()
                appendLine("Input Features:")
                appendLine("  current_speed: ${features[0]}")
                appendLine("  mean_speed: ${features[1]}")
                appendLine("  speed_std: ${features[2]}")
                appendLine("  gyro_z_mean: ${features[3]}")
                appendLine("  gyro_z_max: ${features[4]}")
                appendLine("  acc_y_mean: ${features[5]}")
                appendLine("  acc_y_max: ${features[6]}")
                appendLine("  curve_radius: ${features[7]}")
                appendLine("  severity_proxy: ${features[8]}")
                appendLine()
                appendLine("Probabilities:")
                appendLine("  Safe:   ${(probabilities[0] * 100).toInt()}%")
                appendLine("  Mild:   ${(probabilities[1] * 100).toInt()}%")
                appendLine("  Urgent: ${(probabilities[2] * 100).toInt()}%")
                appendLine("  Hectic: ${(probabilities[3] * 100).toInt()}%")
                appendLine()
                appendLine("Predicted: ${classNames[predictedClass]}")
                appendLine("Confidence: ${(probabilities[predictedClass] * 100).toInt()}%")
            }
            
            resultText.text = result
            
        } catch (e: Exception) {
            resultText.text = "Error: ${e.message}\n\n${e.stackTraceToString()}"
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        modelRunner.close()
    }
}
