import pandas as pd
import numpy as np
import os

df = pd.read_csv("fully_cleaned_features.csv")
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# ==========================================
# 2. CYCLIC TIME INJECTION (Trigonometry)
# ==========================================

# Hour of the day (0-23) -> Divides the circle into 24 slices
df['Hour_Sin'] = np.sin(df.index.hour * (2. * np.pi / 24))
df['Hour_Cos'] = np.cos(df.index.hour * (2. * np.pi / 24))

# Day of the week (0-6) -> Divides the circle into 7 slices
df['Day_Sin'] = np.sin(df.index.dayofweek * (2. * np.pi / 7))
df['Day_Cos'] = np.cos(df.index.dayofweek * (2. * np.pi / 7))

# ==========================================
# 3. CHRONOLOGICAL SPLIT (Train / Val / Test)
# ==========================================
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.8)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

# ==========================================
# 4. STRICT MIN/MAX NORMALIZATION
# ==========================================
# The mathematical anchors - calculated ONLY from the past (Train Set)
min_mw = train_df['PJMW_MW'].min()
max_mw = train_df['PJMW_MW'].max()

def normalize(series, min_v, max_v):
    return (series - min_v) / (max_v - min_v)

# Apply normalization to the Megawatt columns across all sets
for dataset in [train_df, val_df, test_df]:
    dataset['PJMW_MW_Scaled'] = normalize(dataset['PJMW_MW'], min_mw, max_mw)
    dataset['MW_Lag_168_Scaled'] = normalize(dataset['MW_Lag_168'], min_mw, max_mw)

# ==========================================
# 5. ADVANCED SLIDING WINDOW MATRICES
# ==========================================

def create_sliding_window(data, window_size=24):
    X, y = [], []
    
    # Convert columns to fast numpy arrays
    mw_scaled = data['PJMW_MW_Scaled'].values
    lag_scaled = data['MW_Lag_168_Scaled'].values
    h_sin = data['Hour_Sin'].values
    h_cos = data['Hour_Cos'].values
    d_sin = data['Day_Sin'].values
    d_cos = data['Day_Cos'].values

    for i in range(len(data) - window_size):
        # 1. The past 24 hours of grid load (Features 1-24)
        past_24 = mw_scaled[i : i + window_size]
        
        # 2. Context features for the target hour (Features 25-29)
        target_lag = lag_scaled[i + window_size]
        target_h_sin = h_sin[i + window_size]
        target_h_cos = h_cos[i + window_size]
        target_d_sin = d_sin[i + window_size]
        target_d_cos = d_cos[i + window_size]
        
        # 3. Glue them all together into one massive 29-value vector
        x_row = np.concatenate([
            past_24, 
            [target_lag, target_h_sin, target_h_cos, target_d_sin, target_d_cos]
        ])
        
        # 4. The actual answer we want to predict
        y_target = mw_scaled[i + window_size]
        
        X.append(x_row)
        y.append(y_target)
        
    return np.array(X), np.array(y)

X_train, y_train = create_sliding_window(train_df)
X_val, y_val = create_sliding_window(val_df)
X_test, y_test = create_sliding_window(test_df)

print(f"X_train shape is {X_train.shape}")

# ==========================================
# 6. SAVE EVERYTHING TO DISK
# ==========================================
os.makedirs("processed_data", exist_ok=True)

np.savez("processed_data/dataset.npz", 
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test)

np.savez("processed_data/scaling_params.npz", 
         min_value=min_mw, max_value=max_mw)

print("Saved the processed datasets.")