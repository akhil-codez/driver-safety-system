package com.example.driverwarning

import android.content.Context
import android.os.Environment
import android.util.Log
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Logs sensor data to CSV files for training real-world models.
 */
class DataLogger(private val context: Context) {

    private var currentSessionFile: File? = null
    private var fileWriter: FileWriter? = null
    private var isRecording = false
    
    companion object {
        private const val TAG = "DataLogger"
        private const val HEADER = "timestamp,lat,lon,speed,bearing,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,prediction,confidence\n"
    }

    fun startLogging() {
        if (isRecording) return

        try {
            // Use app-specific storage (no permissions needed)
            // Path: Android/data/com.example.driverwarning/files/Documents/DriverWarning
            val appDir = File(context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "DriverWarning")
            if (!appDir.exists()) {
                appDir.mkdirs()
            }

            // Create timestamped file
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            currentSessionFile = File(appDir, "drive_session_$timestamp.csv")
            
            fileWriter = FileWriter(currentSessionFile)
            fileWriter?.append(HEADER)
            
            isRecording = true
            Log.d(TAG, "Started logging to: ${currentSessionFile?.absolutePath}")
            
            // Show toast so user knows where data is
            android.os.Handler(android.os.Looper.getMainLooper()).post {
                android.widget.Toast.makeText(
                    context, 
                    "📂 Recording to internal storage", 
                    android.widget.Toast.LENGTH_SHORT
                ).show()
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error starting logger", e)
        }
    }
    
    /**
     * Get the last modified log file
     */
    fun getLastLogFile(): File? {
        val appDir = File(context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "DriverWarning")
        if (!appDir.exists()) return null
        
        return appDir.listFiles()
            ?.filter { it.name.endsWith(".csv") }
            ?.maxByOrNull { it.lastModified() }
    }

    fun logSample(sample: SensorSample, prediction: String, confidence: Float) {
        if (!isRecording || fileWriter == null) return

        try {
            val line = StringBuilder()
                .append(sample.timestamp).append(",")
                .append(sample.lat).append(",")
                .append(sample.lon).append(",")
                .append(sample.speed).append(",")
                .append(sample.heading).append(",")
                .append(sample.accX).append(",")
                .append(sample.accY).append(",")
                .append(sample.accZ).append(",")
                .append(sample.gyroX).append(",")
                .append(sample.gyroY).append(",")
                .append(sample.gyroZ).append(",")
                .append(prediction).append(",")
                .append(confidence).append("\n")
            
            fileWriter?.append(line.toString())
            
        } catch (e: Exception) {
            Log.e(TAG, "Error logging sample", e)
        }
    }

    fun stopLogging() {
        try {
            fileWriter?.flush()
            fileWriter?.close()
            fileWriter = null
            isRecording = false
            Log.d(TAG, "Stopped logging")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing logger", e)
        }
    }
}
