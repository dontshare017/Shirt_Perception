import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def plot_points_on_image(image, annotations, output_path):
    fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=100)
    ax.imshow(image)

    x_start, y_start = annotations['x_start'], annotations['y_start']
    x_end, y_end = annotations['x_end'], annotations['y_end']

    ax.scatter([x_start, x_end], [y_start, y_end], color='red', s=100)
    ax.plot([x_start, x_end], [y_start, y_end], color='yellow', linewidth=2)
    ax.set_axis_off()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_images_with_annotations(input_dir, output_dir):
    csv_file = None
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            csv_file = os.path.join(input_dir, file)
            break
    
    if not csv_file:
        raise FileNotFoundError("No CSV file found in the directory.")

    annotations = pd.read_csv(csv_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, file)
            output_image_path = os.path.join(output_dir, f"annotated_{file}")

            try:
                image = Image.open(image_path).convert('RGB')
                annotation_row = annotations[annotations['filename'] == file].iloc[0]
                plot_points_on_image(image, annotation_row, output_image_path)
            except Exception as e:
                print(f"Failed to process {file}: {e}")

input_dir = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/rgb/0-30/greyscale_flipped_rgb/T20-T29"
output_dir = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/validation_gt"

process_images_with_annotations(input_dir, output_dir)
