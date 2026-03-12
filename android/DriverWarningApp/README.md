# Driver Warning Android App

Android prototype for the Driver Warning System.

## Setup

1. Open this project in Android Studio
2. Copy `../../models/curve_detector.tflite` to `app/src/main/assets/`
3. Sync Gradle dependencies
4. Build and run on device or emulator

## Requirements

- Android Studio Arctic Fox or later
- Minimum SDK: 24 (Android 7.0)
- Target SDK: 33 (Android 13)
- Kotlin 1.9+

## Key Components

- **MainActivity** - Start/stop service, navigate to debug/settings
- **SensorService** - Background sensor collection and inference
- **TFLiteModelRunner** - TFLite model wrapper
- **FeatureExtractor** - Feature computation (matches Python exactly)
- **AlertManager** - Haptic feedback patterns
- **DebugActivity** - Manual testing with JSON input
- **SettingsActivity** - App configuration

## Testing

### Debug Mode

1. Open app and tap "Debug Testing"
2. Edit the JSON feature values
3. Tap "Run Inference" to see model predictions
4. Test vibration patterns with the test buttons

### Emulator Testing

The app can run on emulator but will not have real sensor data. Use Debug mode to test inference with manual feature input.

### Device Testing

For real testing:
1. Install APK on Android device
2. Grant location permissions
3. Start monitoring
4. Drive or simulate driving motion
5. App will vibrate when detecting curves

## Scaler Parameters

**IMPORTANT**: Before building, update `ScalerParams` in `FeatureExtractor.kt` with actual values from `models/scaler.pkl`.

Extract scaler parameters in Python:
```python
import pickle
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("mean:", scaler.mean_)
print("scale:", scaler.scale_)
```

Then update the arrays in `FeatureExtractor.kt`.

## Build

```bash
./gradlew assembleDebug
```

APK will be in `app/build/outputs/apk/debug/app-debug.apk`

## Known Limitations

- Scaler parameters are placeholder values (need to be updated)
- No Gradle wrapper included (Android Studio will generate)
- Minimal UI styling
- No backend integration for telemetry upload
