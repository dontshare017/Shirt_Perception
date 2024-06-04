import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import numpy as np

def fit_and_compensate_gradient(image_array):

    x = np.arange(image_array.shape[1])
    y = np.arange(image_array.shape[0])
    x, y = np.meshgrid(x, y)
    
    x_flat = x.flatten()
    y_flat = y.flatten()
    image_flat = image_array.flatten()
    
    # polynomial regression
    A = np.vstack([x_flat**2, y_flat**2, x_flat*y_flat, x_flat, y_flat, np.ones(len(x_flat))]).T
    coeff, _, _, _ = np.linalg.lstsq(A, image_flat, rcond=None) # least square fitting

    fitted_surface = (coeff[0] * (x**2) + coeff[1] * (y**2) + coeff[2] * (x*y) + 
                      coeff[3] * x + coeff[4] * y + coeff[5]).reshape(image_array.shape)
    
    adjusted = image_array.astype(np.float32) - fitted_surface

    adjusted -= adjusted.min()  # translate to start at zero
    adjusted /= adjusted.max()  # scale to max of 1
    adjusted *= 255                      # scale to regular rgb range
    return adjusted.astype(np.uint8)

def process_directory(source_dir, target_dir):

    os.makedirs(target_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, rel_path)
        os.makedirs(target_path, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                file_path = os.path.join(root, file)

                image = Image.open(file_path).convert('L')
                image_array = np.array(image)

                compensated_image = fit_and_compensate_gradient(image_array)
                target_file_path = os.path.join(target_path, file)
                Image.fromarray(compensated_image).save(target_file_path)

source_directory = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/depth/60-79/downsampled_depth'
target_directory = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/depth/60-79/downsampled_nograd'
process_directory(source_directory, target_directory)
