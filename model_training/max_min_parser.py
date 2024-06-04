import numpy as np
from PIL import Image
import argparse

def get_depth_image_min_max(image_path):
    try:
        depth_image = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    depth_array = np.array(depth_image)

    min_val = np.min(depth_array)
    max_val = np.max(depth_array)

    return min_val, max_val

if __name__ == '__main__':
    path = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/depth/31-59/downsampled_depth/T31-T39/Depth_188_T37.png"
    min_val, max_val = get_depth_image_min_max(path)

    print(f"Min value in depth image: {min_val}")
    print(f"Max value in depth image: {max_val}")
