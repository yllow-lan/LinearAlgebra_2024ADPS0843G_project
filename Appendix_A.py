import numpy as np
import matplotlib.pyplot as plt

def compute_eigenfaces(image_matrix, k=10):
    """
    Computes the top k Eigenfaces for a given dataset.
    
    Parameters:
    image_matrix (numpy array): A matrix where each COLUMN is a flattened image.
                                Shape: (N^2 pixels, M images)
    k (int): Number of principal components (Eigenfaces) to keep.
    
    Returns:
    eigenfaces (numpy array): The top k Eigenfaces. Shape: (N^2, k)
    weights (numpy array): The weights for each image in the new basis.
    """
    
    # --- Step 1: Compute Mean Face ---
    # Calculate the average pixel value for each position across all images
    mean_face = np.mean(image_matrix, axis=1, keepdims=True)
    
    # --- Step 2: Center the Data ---
    # Subtract the mean face from every image vector to center data at origin
    A = image_matrix - mean_face
    
    # --- Step 3: Compute Covariance Matrix (Surrogate Method) ---
    # We calculate L = A.T * A (size M x M) instead of C = A * A.T (size N^2 x N^2)
    # This is much faster because usually Number of Images (M) < Number of Pixels (N^2)
    L = np.dot(A.T, A)
    
    # --- Step 4: Solve the Eigenvalue Problem (EVP) ---
    # Find eigenvalues (vals) and eigenvectors (vecs) of L
    eigenvalues, eigenvectors = np.linalg.eig(L)
    
    # --- Step 5: Sort Eigenvalues ---
    # Sort in descending order (largest variance first)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # --- Step 6: Map Eigenvectors back to Image Space ---
    # The eigenvectors of L are not the Eigenfaces yet. 
    # We must map them back using: u = A * v
    # We only keep the top k vectors.
    eigenfaces = np.dot(A, eigenvectors[:, :k])
    
    # --- Step 7: Normalize Eigenfaces ---
    # Eigenvectors must be unit length
    for i in range(k):
        eigenfaces[:, i] = eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])
        
    # --- Step 8: Calculate Weights (Compression) ---
    # Project original centered images onto the Eigenfaces
    weights = np.dot(eigenfaces.T, A)
    
    return eigenfaces, weights, mean_face

# --- Example Usage (Simulation) ---
if __name__ == "__main__":
    # 1. Create dummy data: 50 images, each 100x100 pixels (flattened to 10000x1)
    N = 100
    M = 50
    # Random noise + some structure to simulate images
    dummy_data = np.random.rand(N*N, M) 
    
    # 2. Run Eigenfaces algorithm
    k_components = 5
    E, W, mean = compute_eigenfaces(dummy_data, k_components)
    
    print(f"Shape of Input Data: {dummy_data.shape}")
    print(f"Shape of Eigenfaces: {E.shape}") # Should be (10000, 5)
    print(f"Shape of Compressed Weights: {W.shape}") # Should be (5, 50)
    print("Computation Successful.")