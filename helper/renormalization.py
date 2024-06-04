import os
from PIL import Image
import numpy as np

def normalize_depth_images(input_directory):
    for root, dirs, files in os.walk(input_directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.tif', '.tiff')):
                img_path = os.path.join(root, filename)
                with Image.open(img_path) as img:
                    image_array = np.array(img, dtype=np.uint8)
                    min_val = np.min(image_array)
                    max_val = np.max(image_array)
                    if min_val != max_val: 
                        normalized_array = (image_array - min_val) / (max_val - min_val) * 255
                    else:
                        normalized_array = image_array
                    
                    normalized_image = Image.fromarray(normalized_array.astype(np.uint8))

                    normalized_image.save(img_path)

input_dir = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/depth/60-79/downsampled_depth'
normalize_depth_images(input_dir)
