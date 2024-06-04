import os
from PIL import Image

def crop_images(input_directory, output_directory):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            parts = filename.split('_')
            if len(parts) > 2 and parts[2].startswith('T'):
                trial_num_str = parts[2][1:] 
                trial_num = int(trial_num_str.split('.')[0]) 
                trial_group = trial_num // 10 * 10  
                trial_dir = f"T{trial_group}-T{trial_group + 9}"
                # print(trial_dir)
                
                specific_output_dir = os.path.join(output_directory, trial_dir)
                if not os.path.exists(specific_output_dir):
                    os.makedirs(specific_output_dir)

                img_path = os.path.join(input_directory, filename)
                with Image.open(img_path) as img:
                    retain_width = img.width - 1398
                    start_width = 10
                    cropped_img = img.crop((start_width, 0, retain_width, img.height))
                    cropped_img.save(os.path.join(specific_output_dir, filename))

input_dir = '/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/raw_data/raw_rgb_data_61-70'
output_dir = '/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/processed_data/31-60/cropped_rgb'
crop_images(input_dir, output_dir)


input_dir = '/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/raw_data/raw_depth_data_61-70'
output_dir = '/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/processed_data/31-60/cropped_depth'
crop_images(input_dir, output_dir)
