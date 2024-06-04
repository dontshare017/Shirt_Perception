import os
from PIL import Image
import numpy as np

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
                    retain_width = img.width - 1280
                    start_width = 0
                    print(img.mode)
                    cropped = img.crop((start_width, 0, retain_width, img.height))
                    img_arr = np.array(cropped, dtype=np.uint16)
                    min_val = np.min(img_arr)
                    max_val = np.max(img_arr)

                    normalized_arr = ((img_arr - min_val) / (max_val - min_val)) * 65535
                    normalized_img = Image.fromarray(normalized_arr.astype(np.uint16))

                    normalized_img.save(os.path.join(specific_output_dir, filename))



input_dir = '/home/dressing/workspace/dressing_ws/data/6751_shirt_grasping_data/recollection/depth_1-35'
output_dir = '/home/dressing/workspace/dressing_ws/data/6751_shirt_grasping_data/recollection/normalized_depth_1-35'
crop_images(input_dir, output_dir)