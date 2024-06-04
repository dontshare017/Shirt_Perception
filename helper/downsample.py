import os
from PIL import Image
import pandas as pd

def downsample_image(image_path, output_path, downscale_factor=4):
    with Image.open(image_path) as img:
        new_size = (img.width // downscale_factor, img.height // downscale_factor)
        img_resized = img.resize(new_size, Image.LANCZOS)
        img_resized.save(output_path)

def update_annotations(df, downscale_factor=4):
    for column in ['x_start', 'x_end', 'y_start', 'y_end']:
        df[column] = (df[column] // downscale_factor).astype(int)
    return df

def process_images_and_annotations(input_dir, output_dir, annotation_dir, downscale_factor=4):
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                img_path = os.path.join(root, filename)
                output_image_path = os.path.join(output_path, filename)

                downsample_image(img_path, output_image_path)

                trial_dir_name = os.path.basename(root)
                trial_number = trial_dir_name.split('-')[0][1:]

                annotation_csv = os.path.join(annotation_dir, f'annotation_T{trial_number}.csv')
                if os.path.exists(annotation_csv):
                    annotations = pd.read_csv(annotation_csv)
                    updated_annotations = update_annotations(annotations, downscale_factor)

                    updated_csv_path = os.path.join(output_path, f'updated_annotation_T{trial_number}.csv')
                    updated_annotations.to_csv(updated_csv_path, index=False)

input_rgb_dir = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/processed_data/60-79/cropped_rgb'
output_rgb_dir = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/rgb/60-79/downsampled_rgb'
annotation_dir_rgb = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/processed_data/60-79/annotations'

input_depth_dir = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/processed_data/60-79/cropped_depth'
output_depth_dir = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/depth/60-79/downsampled_depth'
annotation_dir_depth = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/processed_data/60-79/annotations'

process_images_and_annotations(input_rgb_dir, output_rgb_dir, annotation_dir_rgb)
process_images_and_annotations(input_depth_dir, output_depth_dir, annotation_dir_depth)
