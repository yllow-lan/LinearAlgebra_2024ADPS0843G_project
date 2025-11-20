import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder_path, image_size=(100, 100)):
    """
    Loads images, converts to grayscale, resizes, and flattens them.
    
    Parameters:
    folder_path (str): Path to the folder containing images.
    image_size (tuple): Target size (width, height) to resize all images.
    
    Returns:
    image_matrix (numpy array): Matrix where each COLUMN is an image.
    """
    image_vectors = []
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(folder_path, filename)
            
            # 1. Open image
            img = Image.open(path)
            
            # 2. Convert to Grayscale ('L' mode)
            # We do this because EVP is usually done on intensity values (1 channel)
            img = img.convert('L')
            
            # 3. Resize
            # All images MUST be the exact same dimensions to form a matrix
            img = img.resize(image_size)
            
            # 4. Flatten
            # Convert 2D image (100x100) to 1D array (10000,)
            img_vector = np.array(img).flatten()
            
            # Add to our list
            image_vectors.append(img_vector)
            
    # 5. Stack vectors as columns
    # If we have 50 images, we get a (10000, 50) matrix
    image_matrix = np.column_stack(image_vectors)
    
    return image_matrix

# --- Usage ---
# folder = "C:/Users/Student/Desktop/MyFaceDatabase"
# X = load_images_from_folder(folder)
# Then pass X to compute_eigenfaces(X)