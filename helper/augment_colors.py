import os
from PIL import Image, ImageEnhance, ImageOps
import random

def random_color_transform(image):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))
    if random.random() > 0.5:
        image = ImageOps.colorize(ImageOps.grayscale(image),
                                  (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                                  (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    return image



def process_directory(input_directory, output_directory):

    for root, dirs, files in os.walk(input_directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                relative_path = os.path.relpath(root, input_directory)
                output_path = os.path.join(output_directory, relative_path)

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                image_path = os.path.join(root, filename)
                image = Image.open(image_path)
                transformed_image = random_color_transform(image)

                save_path = os.path.join(output_path, filename)
                transformed_image.save(save_path)
                # print(f"Processed {filename}")

input_dir = "/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/processed_data/60-79/downsampled/downsampled_rgb/T60-T69"
output_dir = "/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/processed_data/60-79/downsampled/downsampled_color_augmented_rgb/T60-T69"
process_directory(input_dir, output_dir)
