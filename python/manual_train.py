import pandas as pd
import numpy as np
import pickle
import glob
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical

# Settings
DATA_DIR = '../data/'
REAL_DATA_PATTERN = DATA_DIR + "*.csv"
MODEL_DIR = '../android/DriverWarningApp/app/src/main/assets/'

def auto_label(row):
    lat_accel = np.sqrt(row['acc_x']**2 + row['acc_y']**2)
    if lat_accel < 2.0: return 0 # Safe
    if lat_accel < 4.0: return 1 # Mild
    if lat_accel < 6.0: return 2 # Urgent
    return 3 # Hectic

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("🚗 Starting Manual Retraining & Validation...")

# 1. Load Real Data
all_files = glob.glob(REAL_DATA_PATTERN)
real_dfs = []

for filename in all_files:
    if "large_training_data" in filename or "sim_trips" in filename:
        continue
    try:
        df = pd.read_csv(filename)
        if 'acc_x' in df.columns:
            real_dfs.append(df)
            print(f"✅ Loaded {filename} ({len(df)} samples)")
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")

if not real_dfs:
    print("⚠️ No real driving data found!")
    full_real_df = pd.DataFrame()
else:
    full_real_df = pd.concat(real_dfs, ignore_index=True)
    full_real_df['label'] = full_real_df.apply(auto_label, axis=1)
    print(f"Total Real Samples: {len(full_real_df)}")

# 2. Combine with Synthetic
try:
    synthetic_df = pd.read_csv(DATA_DIR + "large_training_data.csv")
    print(f"Loaded {len(synthetic_df)} synthetic samples")
    if 'label' not in synthetic_df.columns:
        print("⏳ Auto-labeling synthetic data...")
        synthetic_df['label'] = synthetic_df.apply(auto_label, axis=1)
except:
    print("⚠️ Large training data not found, using small random batch")
    synthetic_df = pd.DataFrame(np.random.randn(1000, 7), columns=['speed','acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z'])
    synthetic_df['label'] = 0

features = ['speed', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

if not full_real_df.empty:
    X_real = full_real_df[features].values
    y_real = full_real_df['label'].values
    X_synth = synthetic_df[features].values
    y_synth = synthetic_df['label'].values
    
    # Stratified Sampling/Mix could be better, but concat is fine for now
    X = np.concatenate([X_synth, X_real])
    y = np.concatenate([y_synth, y_real])
else:
    X = synthetic_df[features].values
    y = synthetic_df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Cross-Validation
print("\n📊 Running 3-Fold Cross-Validation...")
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold_no = 1
train_accs = []
val_accs = []

for train_index, val_index in kfold.split(X_scaled, y):
    X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]
    
    y_train_cat = to_categorical(y_train_fold, num_classes=4)
    y_val_cat = to_categorical(y_val_fold, num_classes=4)
    
    model = create_model()
    # Train less epochs for CV to be fast
    history = model.fit(X_train_fold, y_train_cat, epochs=5, batch_size=32, verbose=0)
    
    train_acc = history.history['accuracy'][-1]
    val_loss, val_acc = model.evaluate(X_val_fold, y_val_cat, verbose=0)
    
    print(f"   Fold {fold_no}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    fold_no += 1

avg_train = np.mean(train_accs)
avg_val = np.mean(val_accs)
print(f"\n📈 Results:")
print(f"   Avg Training Accuracy:   {avg_train:.4f}")
print(f"   Avg Validation Accuracy: {avg_val:.4f}")

gap = avg_train - avg_val
print("\n🩺 Diagnosis:")
if avg_train < 0.6:
    print("   ❌ UNDERFITTING: The model is not learning well. Try larger model or more epochs/features.")
elif gap > 0.05:
    print("   ⚠️ OVERFITTING: Training is much better than validation. Try more Dropout or L2 Regularization.")
else:
    print("   ✅ GOOD FIT: The model generalizes well!")

# 4. Final Training & Export
print(f"\n🚀 Retraining Final Model on ALL {len(X)} samples for Export...")
y_cat = to_categorical(y, num_classes=4)
final_model = create_model()
final_model.fit(X_scaled, y_cat, epochs=15, batch_size=32, verbose=1) # 15 epochs for final

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

with open(MODEL_DIR + 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
tflite_model = converter.convert()

with open(MODEL_DIR + 'curve_detector.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"✅ Scaler & Model saved to {MODEL_DIR}")
