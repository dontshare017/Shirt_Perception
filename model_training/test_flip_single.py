import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def vertical_flip_image(image, annotations):
    width, height = image.size
    flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)

    updated_annotations = annotations.copy()
    updated_annotations['y_start'] = height - annotations['y_start']
    updated_annotations['y_end'] = height - annotations['y_end']

    return flipped_image, updated_annotations

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

def process_image_and_annotations(image_path, csv_path, output_image_path):

    image = Image.open(image_path)
    annotations = pd.read_csv(csv_path)

    image_filename = os.path.basename(image_path)
    annotation_row = annotations[annotations['filename'] == image_filename].iloc[0]
    flipped_image, updated_annotations = vertical_flip_image(image, annotation_row)
    plot_points_on_image(flipped_image, updated_annotations, output_image_path)

input_image_path = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/processed_data/0-30/cropped_rgb/T20-T29/RGB_140_T27.png"
annotation_csv_path = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/processed_data/0-30/cropped_rgb/T20-T29/annotation_T20-T29.csv"
output_image_path = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/validation_gt"

process_image_and_annotations(input_image_path, annotation_csv_path, output_image_path)
