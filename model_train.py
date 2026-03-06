import numpy as np
import os
from scipy.spatial.distance import cdist
from scipy.linalg import solve

# ==========================================
# 1. LOAD DATA & CHAMPION PARAMETERS
# ==========================================
print("Loading Data and Champion Parameters...")
dataset = np.load("processed_data/dataset.npz")
X_train, y_train = dataset['X_train'], dataset['y_train']

# Load the unseen test set to evaluate final performance
X_test, y_test = dataset['X_test'], dataset['y_test'] 

# Load scaling params to calculate real-world error
params = np.load("processed_data/scaling_params.npz")
min_val, max_val = params['min_value'], params['max_value']

centers = np.load("processed_data/best_centers.npy")
sigmas = np.load("processed_data/best_sigmas.npy")

K = len(centers)
print(f"Loaded {len(X_train)} training rows, {len(X_test)} test rows, and {K} centers.")

# ==========================================
# 2. BUILD THE HIDDEN LAYER (RAM OPTIMIZED)
# ==========================================
def calculate_activations(X, centers, sigmas, batch_size=10000):
    """Transforms X into the hidden layer matrix G using batched cdist."""
    G = np.zeros((len(X), len(centers)))
    for i in range(0, len(X), batch_size):
        end = min(i + batch_size, len(X))
        distances = cdist(X[i:end], centers)
        G[i:end] = np.exp(-(distances ** 2) / (2 * sigmas ** 2))
    return G

print("Pushing X_train through the Gaussian hidden layer...")
G_train = calculate_activations(X_train, centers, sigmas)

# ==========================================
# 3. TRAIN THE OUTPUT LAYER (L2 REGULARIZED)
# ==========================================
print("Solving for output weights using optimized Ridge Regression...")

lambda_val = 0  # The penalty strength (tweak this later!)
identity_matrix = np.eye(K)

# OPTIMIZATION: Instead of calculating the inverse matrix (memory heavy & unstable),
# we solve the linear system (A * W = b) directly using SciPy.
A = G_train.T.dot(G_train) + lambda_val * identity_matrix
b = G_train.T.dot(y_train)

# 'assume_a="pos"' speeds it up because A is a positive-definite matrix
weights = solve(A, b, assume_a='pos') 

print(f"Shape of Final Weights: {weights.shape}")

# ==========================================
# 4. EVALUATE ON THE TEST SET
# ==========================================
print("\nEvaluating final model on the unseen Test Set...")

G_test = calculate_activations(X_test, centers, sigmas)
y_test_pred_scaled = G_test.dot(weights)

def inverse_transform(scaled_val, min_v, max_v):
    return (scaled_val * (max_v - min_v)) + min_v

y_test_real_mw = inverse_transform(y_test, min_val, max_val)
y_test_pred_real_mw = inverse_transform(y_test_pred_scaled, min_val, max_val)

test_mae = np.mean(np.abs(y_test_real_mw - y_test_pred_real_mw))
print(f"🔥 FINAL TEST ERROR: ±{test_mae:.2f} Megawatts")

# ==========================================
# 5. SAVE THE FINAL DEPLOYMENT MODEL
# ==========================================
print("\nSaving the finalized model...")

# Note: We save 'sigmas' under the key 'sigma' so your existing AWS Lambda 
# script can load it without needing any code changes!
np.savez("processed_data/rbf_model_opt.npz", 
         centers=centers, 
         sigma=sigmas, 
         weights=weights)

print("🏆 Training complete! 'processed_data/rbf_model_opt.npz' is ready for the cloud.")