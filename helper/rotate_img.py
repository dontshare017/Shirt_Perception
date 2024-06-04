import os
import pandas as pd
from PIL import Image
import random
import math

def rotate_images(rgb_image, depth_image, angle, annotations):
    width, height = rgb_image.size
    angle_rad = math.radians(angle)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    cx, cy = width // 2, height // 2

    new_annotations = annotations.copy()
    for index, row in annotations.iterrows():
        x_start, y_start = row['x_start'] - cx, row['y_start'] - cy
        x_end, y_end = row['x_end'] - cx, row['y_end'] - cy

        new_x_start = cx + (x_start * cos_theta - y_start * sin_theta)
        new_y_start = cy + (x_start * sin_theta + y_start * cos_theta)
        new_x_end = cx + (x_end * cos_theta - y_end * sin_theta)
        new_y_end = cy + (x_end * sin_theta + y_end * cos_theta)

        new_annotations.at[index, 'x_start'] = int(new_x_start)
        new_annotations.at[index, 'y_start'] = int(new_y_start)
        new_annotations.at[index, 'x_end'] = int(new_x_end)
        new_annotations.at[index, 'y_end'] = int(new_y_end)

    fill_color_rgb = 255 if rgb_image.mode == 'L' else (255, 255, 255)
    fill_color_depth = 65535 if depth_image.mode == 'I' or depth_image.mode == 'F' else (255, 255, 255) if depth_image.mode == 'RGB' else 255

    rotated_rgb = rgb_image.rotate(-angle, center=(cx, cy), expand=False, fillcolor=fill_color_rgb)
    rotated_depth = depth_image.rotate(-angle, center=(cx, cy), expand=False, fillcolor=fill_color_depth)

    return rotated_rgb, rotated_depth, new_annotations


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

                relative_path_depth = os.path.relpath(root, rgb_input_dir)
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

                angle = random.uniform(-30, 30)
                rotated_rgb, rotated_depth, updated_annotations = rotate_images(rgb_image, depth_image, angle, annotation_row)

                save_path_depth = os.path.join(output_path_depth, filename.replace("RGB", "Depth"))
                save_path_rgb = os.path.join(output_path_rgb, filename)

                rotated_depth.save(save_path_depth)
                rotated_rgb.save(save_path_rgb)

                if annotations is not None:
                    annotations.loc[annotations['filename'] == filename, ['x_start', 'x_end', 'y_start', 'y_end']] = updated_annotations[['x_start', 'x_end', 'y_start', 'y_end']]

        if csv_file and annotations is not None:
            new_csv_filename = csv_file[0]
            annotations.to_csv(os.path.join(output_path_rgb, new_csv_filename), index=False)
            annotations.to_csv(os.path.join(output_path_depth, new_csv_filename), index=False)


input_dir_rgb = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/rgb/60-79/greyscale_rgb"
output_dir_rgb = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/rgb/60-79/greyscale_rotated_rgb"
input_dir_depth = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/depth/60-79/grayscale_depth"
output_dir_depth = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/depth/60-79/greyscale_rotated_depth"
process_directory(input_dir_rgb, input_dir_depth, output_dir_rgb, output_dir_depth)
