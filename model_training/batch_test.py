import os
import json
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from rgb_dataset_greyscale import RGBDDataset
from coordinate_unet import CoordinateUNet
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from skimage.transform import resize

with open('configs_split.json', 'r') as cfg_file:
    cfgs = json.load(cfg_file)

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = CoordinateUNet()
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model

def plot_image_with_annotations(img_rgb, img_depth, pred_coords, true_coords, image_filename):
    target_resolution = (768, 672)  # h x w
    current_resolution = img_rgb.shape

    if current_resolution != target_resolution:
        img_rgb = resize(img_rgb, target_resolution, anti_aliasing=True)
        img_depth = resize(img_depth, target_resolution, anti_aliasing=True)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # handle grayscale
    for ax, img, title in zip(axs, [img_rgb, img_depth], ['Grayscale Image', 'Depth Image']):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(title)

        scale_x = target_resolution[1] / current_resolution[1]  # scale width
        scale_y = target_resolution[0] / current_resolution[0]  # scale height
        x_start, x_end, y_start, y_end = pred_coords
        x_start, x_end = x_start * scale_x, x_end * scale_x
        y_start, y_end = y_start * scale_y, y_end * scale_y

        true_x_start, true_x_end, true_y_start, true_y_end = true_coords
        true_x_start, true_x_end = true_x_start * scale_x, true_x_end * scale_x
        true_y_start, true_y_end = true_y_start * scale_y, true_y_end * scale_y

        ax.plot([x_start, x_end], [y_start, y_end], 'k-', linewidth=1)
        ax.scatter([x_start, x_end], [y_start, y_end], c='red', edgecolors='black', s=50, label='Prediction')
        ax.plot([true_x_start, true_x_end], [true_y_start, true_y_end], 'k-', linewidth=1)
        ax.scatter([true_x_start, true_x_end], [true_y_start, true_y_end], c='green', edgecolors='black', s=50, label='Ground Truth')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    save_path = os.path.join("/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/validation_3", image_filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def test_model(model, device, test_loader):
    idx = 0
    with torch.no_grad():
        for data in test_loader:
            images, true_coords = data['image'].to(device), data['coords'].cpu().numpy()
            outputs = model(images).cpu().numpy()
            images_np = images.cpu().numpy()
            
            # for img, output, true_coord in zip(images_np, outputs, true_coords):
            #     # Assuming the first three channels are RGB and fourth channel is depth
            #     img_rgb = np.transpose(img[:3], (1, 2, 0))
            #     img_depth = img[3]
            #     plot_image_with_annotations(img_rgb, img_depth, output, true_coord, f'{idx}.png')
            #     idx += 1

            for img, output, true_coord in zip(images_np, outputs, true_coords):
                # Assuming the first channel is grayscale and the second is depth
                img_rgb = img[0]  # Grayscale
                img_depth = img[1]  # Depth
                plot_image_with_annotations(img_rgb, img_depth, output, true_coord, f'{idx}.png')
                idx += 1


def prepare_datasets(cfgs):
    unique_indices = range(814)
    train_indices, test_indices = train_test_split(unique_indices, test_size=0.2, random_state=42)

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = RGBDDataset(
        root_rgb_dir=cfgs["rgb_root_dir"],
        root_depth_dir=cfgs["depth_root_dir"],
        transform=transform,
        indices=list(test_indices)
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    return test_loader

if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = cfgs['model_path']
    test_loader = prepare_datasets(cfgs)
    model = load_model(model_path, device)
    test_model(model, device, test_loader)
