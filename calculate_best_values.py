import numpy as np
import os
from scipy.spatial.distance import cdist
from scipy.linalg import lstsq

# ==========================================
# 1. LOAD DATA & SCALING PARAMS
# ==========================================
dataset = np.load("processed_data/dataset.npz")
X_train, y_train = dataset['X_train'], dataset['y_train']
X_val, y_val = dataset['X_val'], dataset['y_val']

params = np.load("processed_data/scaling_params.npz")
min_val, max_val = params['min_value'], params['max_value']

P_NEIGHBORS = 15

def inverse_transform(scaled_val, min_v, max_v):
    """Reverses the Min-Max scaling to get real Megawatts."""
    return (scaled_val * (max_v - min_v)) + min_v

y_val_real_mw = inverse_transform(y_val, min_val, max_val)

# ==========================================
# 2. THE PURE MATH ENGINE (RAM OPTIMIZED)
# ==========================================
def get_rbf_centers(X, K, max_iters=50, tolerance=1e-4, batch_size=10000):
    """K-Means Clustering implemented with memory-efficient SciPy cdist."""
    np.random.seed(42)
    # 1. Randomly pick K data points to be our starting centers
    centers = X[np.random.choice(len(X), size=K, replace=False)]
    
    for _ in range(max_iters):
        # 2 & 3. Process distances and assign labels in batches to save RAM
        labels = np.zeros(len(X), dtype=int)
        for i in range(0, len(X), batch_size):
            end = min(i + batch_size, len(X))
            # cdist computes the distance matrix without the 3D broadcasting overhead
            batch_dist = cdist(X[i:end], centers)
            labels[i:end] = np.argmin(batch_dist, axis=1)
        
        # 4. Move the center to the exact average of all its assigned points
        new_centers = np.array([X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 
                                else X[np.random.choice(len(X))] for j in range(K)])
        
        # 5. Stop if the centers have stopped moving
        if np.linalg.norm(new_centers - centers) < tolerance:
            break
        centers = new_centers
        
    return centers

def calculate_local_sigmas(centers, p_neighbors=P_NEIGHBORS):
    """Calculates a unique width for each center based on nearest neighbors."""
    K = len(centers)
    sigmas = np.zeros(K)
    
    # Calculate distance between all centers
    distances = cdist(centers, centers)
    
    for i in range(K):
        # Sort distances and grab the P nearest neighbors (skipping index 0, which is itself)
        nearest_distances = np.sort(distances[i])[1 : p_neighbors + 1]
        sigmas[i] = np.mean(nearest_distances)
        if sigmas[i] == 0:
            sigmas[i] = 1e-8
            
    return sigmas

def calculate_activations(X, centers, sigmas, batch_size=10000):
    """Transforms X into the hidden layer G using memory-efficient batching."""
    G = np.zeros((len(X), len(centers)))
    
    # Process activations in batches to prevent OOM
    for i in range(0, len(X), batch_size):
        end = min(i + batch_size, len(X))
        distances = cdist(X[i:end], centers)
        # NumPy broadcasts the sigmas array perfectly across the columns
        G[i:end] = np.exp(-(distances ** 2) / (2 * sigmas ** 2))
        
    return G

# ==========================================
# 3. THE VALIDATION ARENA (Grid Search)
# ==========================================
K_list = [2, 500]

best_K = None
lowest_error = float('inf')
best_centers = None
best_sigmas = None


for K in K_list:
    print(f"\nTraining model with K={K}...")
    
    # 1. Find Centers
    centers = get_rbf_centers(X_train, K)
    
    # 2. Calculate Local Sigmas
    sigmas = calculate_local_sigmas(centers, p_neighbors=P_NEIGHBORS)
    
    # 3. Train Weights (Using SciPy lstsq instead of pinv for memory efficiency)
    G_train = calculate_activations(X_train, centers, sigmas)
    weights, _, _, _ = lstsq(G_train, y_train, lapack_driver='gelsd')
    
    # 4. Test on Validation Set
    G_val = calculate_activations(X_val, centers, sigmas)
    y_val_pred_scaled = G_val.dot(weights)
    
    # 5. Calculate Real-World Error
    y_val_pred_real_mw = inverse_transform(y_val_pred_scaled, min_val, max_val)
    mae = np.mean(np.abs(y_val_real_mw - y_val_pred_real_mw))
    
    print(f"Result for K={K}: Average Error of {mae:.2f} Megawatts")
    
    # 6. Track the champion
    if mae < lowest_error:
        lowest_error = mae
        best_K = K
        best_centers = centers
        best_sigmas = sigmas

# ==========================================
# 4. SAVE THE CHAMPION
# ==========================================
print(f"Best value: K = {best_K} with {lowest_error:.2f} MW error")

os.makedirs("processed_data", exist_ok=True)
np.save("processed_data/best_centers.npy", best_centers)
np.save("processed_data/best_sigmas.npy", best_sigmas)

print(f"Saved {best_K}")