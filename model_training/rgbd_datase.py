import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import os

class RGBDDataset(Dataset):
    def __init__(self, root_rgb_dir, root_depth_dir, transform=None):
        self.rgb_root_dir = root_rgb_dir
        self.depth_root_dir = root_depth_dir
        self.transform = transform
        self.data = []
        self.load_data()

    def load_data(self):
        depth_subfolder = 'downsampled_nomalized_depth'
        for trial_range in ['0-30', '31-59', '60-79']:
            rgb_trial_path = os.path.join(self.rgb_root_dir, trial_range)
            depth_trial_path = os.path.join(self.depth_root_dir, trial_range)

            for subdir in os.listdir(rgb_trial_path):
                subdir_path = os.path.join(rgb_trial_path, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                for child_dir in os.listdir(subdir_path):
                    child_dir_path = os.path.join(subdir_path, child_dir)
                    print(child_dir_path)
                    if not os.path.isdir(child_dir_path):
                        continue

                    csv_file_path = self.find_csv_file(child_dir_path)
                    if csv_file_path:
                        annotations = pd.read_csv(csv_file_path)
                        for index, row in annotations.iterrows():
                            rgb_img_path = os.path.join(child_dir_path, row['filename'])
                            depth_img_path = os.path.join(depth_trial_path, depth_subfolder, child_dir, row['filename'].replace("RGB", "Depth"))

                            if os.path.exists(rgb_img_path) and os.path.exists(depth_img_path):
                                self.data.append({
                                    'rgb_img_path': rgb_img_path,
                                    'depth_img_path': depth_img_path,
                                    'coords': row[['x_start', 'x_end', 'y_start', 'y_end']].values.astype(np.int64)
                                })

    def find_csv_file(self, child_dir_path):
        for file in os.listdir(child_dir_path):
            if file.endswith('.csv'):
                return os.path.join(child_dir_path, file)
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        try:
            rgb_image = Image.open(sample['rgb_img_path']).convert('RGB')
            depth_image = Image.open(sample['depth_img_path']).convert('L')
            image = np.concatenate((np.array(rgb_image), np.expand_dims(np.array(depth_image), axis=2)), axis=2)

            if self.transform:
                image = self.transform(image)

            return {'image': image, 'coords': sample['coords']}

        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            raise

# dataset = RGBDDataset(root_rgb_dir="path/to/rgb_root", root_depth_dir="path/to/depth_root", transform=your_transforms_here)
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True)