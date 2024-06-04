import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from modified_unet import ModifiedUNet
import matplotlib.patches as patches

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = ModifiedUNet()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def preprocess_and_load_image(rgb_path, depth_path):
    rgb_image = Image.open(rgb_path).convert('RGB')
    depth_image = Image.open(depth_path).convert('L')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    rgb_tensor = transform(rgb_image)
    depth_tensor = transform(depth_image) 

    image_tensor = torch.cat((rgb_tensor, depth_tensor), dim=0) 

    return image_tensor.unsqueeze(0)


def plot_image_with_annotations(image_tensor, coords):
    image_np = image_tensor.numpy().transpose(1, 2, 0)
    image_rgb = image_np[:, :, :3] 

    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    x_start, x_end, y_start, y_end = coords
    ax.scatter([x_start, x_end], [y_start, y_end], c='red', edgecolors='black', s=50, zorder=3, label='Prediction')
    ax.axis('off')
    plt.show()

def main(model_path, rgb_path, depth_path):
    model = load_model(model_path)
    image_tensor = preprocess_and_load_image(rgb_path, depth_path)
    with torch.no_grad():
        outputs = model(image_tensor).cpu().numpy().flatten() 
    plot_image_with_annotations(image_tensor[0], outputs)

if __name__ == "__main__":
    model_path = '/Users/YifeiHu/Downloads/model/final_run/model_epoch_106_20240514_034231.pth' 
    image_path = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/color_aug_rgb/0-30/downsampled_color_augmented_rgb/T20-T29/RGB_124_T23.png'  
    depth_path = '/Users/YifeiHu/Desktop/Cornell/Emprise_Lab/Shirt_Perception/downsampled_final/color_aug_depth/0-30/downsampled_color_augmented_depth/T20-T29/Depth_124_T23.png'
    main(model_path, image_path, depth_path)
