import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import glob
from sklearn.metrics import classification_report, confusion_matrix

# Paths
DATA_DIR = '../data/'
REAL_DATA_PATTERN = DATA_DIR + "*.csv"
ASSETS_DIR = '../android/DriverWarningApp/app/src/main/assets/'
MODEL_PATH = ASSETS_DIR + 'curve_detector.tflite'
SCALER_PATH = ASSETS_DIR + 'scaler.pkl'

def auto_label(row):
    lat_accel = np.sqrt(row['acc_x']**2 + row['acc_y']**2)
    if lat_accel < 2.0: return 0
    if lat_accel < 4.0: return 1
    if lat_accel < 6.0: return 2
    return 3

print("🔍 Testing TFLite Model Accuracy...")

# 1. Load Data
all_files = glob.glob(REAL_DATA_PATTERN)
real_dfs = []
for filename in all_files:
    if "large_training_data" in filename or "sim_trips" in filename: continue
    try:
        df = pd.read_csv(filename)
        if 'acc_x' in df.columns: real_dfs.append(df)
    except: pass

if not real_dfs:
    print("❌ No real data found to test against.")
    exit()

df = pd.concat(real_dfs, ignore_index=True)
df['ground_truth'] = df.apply(auto_label, axis=1)

features = ['speed', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
X_raw = df[features].values
y_true = df['ground_truth'].values

print(f"📊 Loaded {len(X_raw)} real world samples.")

# 2. Load Scaler & Model
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded.")
    
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("✅ TFLite Model loaded.")
except Exception as e:
    print(f"❌ Error loading assets: {e}")
    exit()

# 3. Predict (Simulate App Logic)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']

X_scaled = scaler.transform(X_raw)
y_pred = []

for i in range(len(X_scaled)):
    input_data = np.array([X_scaled[i]], dtype=np.float32)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)
    predicted_class = np.argmax(output_data)
    y_pred.append(predicted_class)

# 4. Report
print("\n📝 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Safe', 'Mild', 'Urgent', 'Hectic'], labels=[0,1,2,3], zero_division=0))

print("\nBxB Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Check specific failure Mode
print("\n🕵️ Logic Check:")
safe_samples = df[df['ground_truth'] == 0]
if not safe_samples.empty:
    X_safe = scaler.transform(safe_samples[features].values)
    # just check first one
    input_data = np.array([X_safe[0]], dtype=np.float32)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    out = interpreter.get_tensor(output_index)
    print(f"Example SAFE sample inputs: {safe_samples.iloc[0][features].values}")
    print(f"Model Probability Output: {out}")
    print(f"Predicted: {np.argmax(out)} (Should be 0)")
