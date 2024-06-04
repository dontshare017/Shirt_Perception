import os
import pandas as pd
from PIL import Image
import random

def random_flip_image(rgb, depth, annotations):
    width, height = rgb.size
    updated_annotations = annotations.copy()

    if random.random() > 0.5:
        rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        updated_annotations['x_start'] = width - updated_annotations['x_start'] 
        updated_annotations['x_end'] = width - updated_annotations['x_end']    

    if random.random() > 0.5:
        rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM)
        depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
        updated_annotations['y_start'] = height - updated_annotations['y_start'] 
        updated_annotations['y_end'] = height - updated_annotations['y_end']    

    return rgb, depth, updated_annotations


def process_directory(rgb_input_dir, depth_input_dir, rgb_output_dir, depth_output_dir):

    for root, dirs, files in os.walk(rgb_input_dir):
        csv_file = [f for f in files if f.endswith('.csv')]
        annotations = None
        if csv_file:
            csv_path = os.path.join(root, csv_file[0])
            annotations = pd.read_csv(csv_path)
        
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                relative_path_rgb = os.path.relpath(root, rgb_input_dir)
                output_path_rgb = os.path.join(rgb_output_dir, relative_path_rgb)

                depth_image_dir = os.path.join(depth_input_dir, relative_path_rgb)
                output_path_depth = os.path.join(depth_output_dir, relative_path_rgb)

                if not os.path.exists(output_path_rgb):
                    os.makedirs(output_path_rgb)
                if not os.path.exists(output_path_depth):
                    os.makedirs(output_path_depth)

                rgb_image_path = os.path.join(root, filename)
                rgb_image = Image.open(rgb_image_path)
                
                depth_image_path = os.path.join(depth_image_dir, filename.replace("RGB", "Depth"))
                if not os.path.exists(depth_image_path):
                    continue

                depth_image = Image.open(depth_image_path)
                if annotations is not None:
                    annotation_row = annotations[annotations['filename'] == filename]
                else:
                    annotation_row = pd.DataFrame()

                flipped_rgb, flipped_depth, updated_annotations = random_flip_image(rgb_image, depth_image, annotation_row)
                save_path_depth = os.path.join(output_path_depth, filename.replace("RGB", "Depth"))
                save_path_rgb = os.path.join(output_path_rgb, filename)

                flipped_depth.save(save_path_depth)
                flipped_rgb.save(save_path_rgb)

                if annotations is not None:
                    annotations.loc[annotations['filename'] == filename, ['x_start', 'x_end', 'y_start', 'y_end']] = updated_annotations[['x_start', 'x_end', 'y_start', 'y_end']]

        if csv_file and annotations is not None:
            new_csv_filename = csv_file[0]
            annotations.to_csv(os.path.join(output_path_rgb, new_csv_filename), index=False)
            annotations.to_csv(os.path.join(output_path_depth, new_csv_filename), index=False)

input_dir_rgb = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/rgb/0-30/greyscale_rgb"
output_dir_rgb = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/rgb/0-30/greyscale_flipped_rgb"
input_dir_depth = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/depth/0-30/grayscale_depth"
output_dir_depth = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/depth/0-30/greyscale_flipped_depth"
process_directory(input_dir_rgb, input_dir_depth, output_dir_rgb, output_dir_depth)
