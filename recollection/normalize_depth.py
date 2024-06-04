import numpy as np
from PIL import Image

def normalize(input_path, output_path):
    with Image.open(input_path) as img:
        print(img.mode)
        retain_width = img.width - 1398
        start_width = 10
        cropped = img.crop((start_width, 0, retain_width, img.height))
        img_arr = np.array(cropped, dtype=np.uint16)
        min_val = np.min(img_arr)
        max_val = np.max(img_arr)

        normalized_arr = ((img_arr - min_val) / (max_val - min_val)) * 65535
        normalized_img = Image.fromarray(normalized_arr.astype(np.uint16))

        normalized_img.save(output_path)

def normalize_w_img(img, output_path):
    print(img.mode)
    retain_width = img.width - 1398 # make it 1280
    start_width = 10
    cropped = img.crop((start_width, 0, retain_width, img.height))
    img_arr = np.array(cropped, dtype=np.uint16)
    min_val = np.min(img_arr)
    max_val = np.max(img_arr)

    normalized_arr = ((img_arr - min_val) / (max_val - min_val)) * 65535
    normalized_img = Image.fromarray(normalized_arr.astype(np.uint16))

    normalized_img.save(output_path)






input_path = "/home/dressing/workspace/dressing_ws/data/6751_shirt_grasping_data/raw_depth_data_71/Depth_351_T71.png"
output_path = "/home/dressing/workspace/dressing_ws/data/6751_shirt_grasping_data/raw_depth_data_71/Depth_351_T71_processed.png"

normalize(input_path, output_path)