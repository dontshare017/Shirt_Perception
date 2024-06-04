import os
from PIL import Image

def convert_to_grayscale(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, rel_path)
        
        os.makedirs(target_path, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:
                    grayscale_img = img.convert('L')
                    target_file_path = os.path.join(target_path, file)
                    grayscale_img.save(target_file_path)

source_directory = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/rgb/60-79/downsampled_rgb'
target_directory = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/rgb/60-79/downsampled_greyscale'
convert_to_grayscale(source_directory, target_directory)
