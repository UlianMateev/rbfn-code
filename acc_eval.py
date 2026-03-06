import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD THE TEST DATA & MODEL
# ==========================================
print("Loading unseen Test Data and finalized Model...")

# Load the exact Test Set your model has never seen
dataset = np.load("processed_data/dataset.npz")
X_test = dataset['X_test']
y_test = dataset['y_test']

# Load your mathematically perfect parameters
model = np.load("processed_data/rbf_model_opt.npz")
centers = model['centers']
sigmas = model['sigma']  # This is now your array of Local Sigmas!
weights = model['weights']

# Load the anchors to convert back to Megawatts
params = np.load("processed_data/scaling_params.npz")
min_val = params['min_value']
max_val = params['max_value']

print(f"Loaded {len(X_test)} hours of test data with {X_test.shape[1]} features each.")

# ==========================================
# 2. THE PREDICTION MATH (RAM OPTIMIZED)
# ==========================================
def calculate_activations(X, centers, sigmas, batch_size=10000):
    """Transforms inputs into Gaussian activations using RAM-safe batched cdist."""
    G = np.zeros((len(X), len(centers)))
    for i in range(0, len(X), batch_size):
        end = min(i + batch_size, len(X))
        distances = cdist(X[i:end], centers)
        G[i:end] = np.exp(-(distances ** 2) / (2 * sigmas ** 2))
    return G

def inverse_transform(scaled_val, min_v, max_v):
    """Reverses the Min-Max scaling back to real Megawatts."""
    return (scaled_val * (max_v - min_v)) + min_v

print("Running predictions on the Test Set...")

# 1. Pass the test data through the hidden layer
G_test = calculate_activations(X_test, centers, sigmas)

# 2. Multiply by the weights to get the scaled predictions
y_pred_scaled = G_test.dot(weights)

# 3. Convert both predictions and actual answers back to real Megawatts
y_pred_real_mw = inverse_transform(y_pred_scaled, min_val, max_val)
y_test_real_mw = inverse_transform(y_test, min_val, max_val)

# ==========================================
# 3. CALCULATE DEEP METRICS
# ==========================================
errors = y_test_real_mw - y_pred_real_mw
abs_errors = np.abs(errors)

# 1. Standard Errors
mae = np.mean(abs_errors)
rmse = np.sqrt(np.mean(errors ** 2))

# 2. Percentage Error (Added a tiny epsilon to prevent divide-by-zero just in case)
mape = np.mean(abs_errors / (y_test_real_mw + 1e-8)) * 100

# 3. R-squared (R2) Score
ss_res = np.sum(errors ** 2)
ss_tot = np.sum((y_test_real_mw - np.mean(y_test_real_mw)) ** 2)
r2_score = 1 - (ss_res / ss_tot)

# 4. Outlier Analysis
max_error = np.max(abs_errors)
medae = np.median(abs_errors)

print("==========================================")
print(f"R-Squared (R²):          {r2_score:.4f}")
print(f"Average Error (MAE):     ±{mae:.2f} MW")
print(f"Median Error (MedAE):    ±{medae:.2f} MW")
print(f"Worst-Case Error (RMSE): ±{rmse:.2f} MW")
print(f"Max Single Miss:         ±{max_error:.2f} MW")
print(f"Percentage Error (MAPE): {mape:.2f}%")
print("==========================================")

# ==========================================
# 4. VISUALIZATION FOR DOCUMENTATION
# ==========================================
print("\nGenerating graph for documentation...")

# We plot a 2-week slice (24 hours * 14 days = 336 hours) so the lines are clearly visible.
# You can change this number if you want to zoom in or out!
hours_to_plot = 336 

plt.figure(figsize=(14, 6))

# Plot the real data as a solid black line
plt.plot(y_test_real_mw[:hours_to_plot], label='Реално Натоварване', color='black', linewidth=1.5)

# Plot the model's predictions as a dashed red line
plt.plot(y_pred_real_mw[:hours_to_plot], label='RBFN Предвидени', color='red', linestyle='--', linewidth=1.5)

plt.title('RBFN: Реални срещу Предвидени стойности в мегаватове (2-седмици)', fontsize=14, fontweight='bold')
plt.xlabel('Time (Hours)', fontsize=12)
plt.ylabel('Energy Load (Megawatts)', fontsize=12)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)

# Make the layout tight so it looks clean when you screenshot it
plt.tight_layout()
plt.show()