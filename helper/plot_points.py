import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def plot_points_on_images(input_directory, output_directory):

    csv_file = next((f for f in os.listdir(input_directory) if f.endswith('.csv')), None)
    if csv_file is None:
        raise FileNotFoundError("No CSV file found in the directory.")

    csv_path = os.path.join(input_directory, csv_file)
    annotations = pd.read_csv(csv_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    for index, row in annotations.iterrows():
        image_filename = row['filename'].replace("RGB", "Depth")
        image_path = os.path.join(input_directory, image_filename)
        
        if os.path.exists(image_path):
            image = Image.open(image_path)
            dpi = 100
            fig_width, fig_height = image.width / dpi, image.height / dpi
            # fig_width, fig_height = image.width, image.height
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            ax.imshow(image)
            
            x_start, y_start = row['x_start'], row['y_start']
            x_end, y_end = row['x_end'], row['y_end']
            
            ax.scatter([x_start, x_end], [y_start, y_end], color='red', s=100)  # s is the size of the point
            ax.plot([x_start, x_end], [y_start, y_end], color='yellow', linewidth=2)

            ax.set_axis_off()

            save_path = os.path.join(output_directory, image_filename)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        
            print(f"Processed and saved: {save_path}")
        else:
            print(f"Image file not found: {image_filename}")



input_dir = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/depth/0-30/downsampled_flipped_depth/T20-T29"
output_dir = "/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/validation"
plot_points_on_images(input_dir, output_dir)
