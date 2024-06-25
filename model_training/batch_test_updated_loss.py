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

#with open('configs_split.json', 'r') as cfg_file:
with open('configs_split2.json', 'r') as cfg_file:
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

        true_x_start, true_x_end, true_y_start, true_y_end, radius = true_coords
        true_x_start, true_x_end = true_x_start * scale_x, true_x_end * scale_x
        true_y_start, true_y_end = true_y_start * scale_y, true_y_end * scale_y

        ax.plot([x_start, x_end], [y_start, y_end], 'k-', linewidth=1)
        ax.scatter(x_start, y_start, c='red', edgecolors='black', s=50, label='Prediction' , marker="*")
        ax.scatter(x_end,  y_end, c='red', edgecolors='black', s=50, label='Prediction')
        #ax.scatter([x_start, x_end], [y_start, y_end], c='red', edgecolors='black', s=50, label='Prediction')
        # ax.plot([true_x_start, true_x_end], [true_y_start, true_y_end], 'k-', linewidth=1)
        # ax.scatter([true_x_start, true_x_end], [true_y_start, true_y_end], c='green', edgecolors='black', s=50, label='Ground Truth')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    print("saving ")
    save_path = os.path.join("C:/Users/ACER/Downloads/greyscale_data_modified_annotations/model_predictions", image_filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def loss(outputs, labels):
    center_x_label = labels[:,0]
    center_y_label = labels[:,2]
    end_x_label = labels[:,1]
    end_y_label  = labels[:,3]
    radius_list = labels[:,4]

    print(radius_list)

    labels = labels[:,:4]
    #print("shape is ",labels.shape)
    center_x_output = outputs[:,0]
    center_y_output = outputs[:,2]
    end_x_output = outputs[:,1]
    end_y_output  = outputs[:,3]

    distance_list = torch.sqrt((center_x_label - center_x_output)**2 + (center_y_label - center_y_output)**2)
    diff_list_dist = distance_list - radius_list
    diff_list_dist = torch.clamp(diff_list_dist, min=0)
    angle_list_outputs = torch.atan2((end_y_output-center_y_output),(end_x_output-center_x_output))*180/(torch.pi)
    angle_list_labels = torch.atan2((end_y_label-center_y_label),(end_x_label-center_x_label))*180/(torch.pi)
    diff_list_angle = torch.abs(angle_list_labels - angle_list_outputs)

    loss_val = 100*torch.mean(diff_list_dist) + 100*torch.mean(diff_list_angle)
    return loss_val



def test_model(model, device, test_loader):
    idx = 0
    with torch.no_grad():
        for data in test_loader:
            images, true_coords = data['image'].to(device), data['coords'].cpu().numpy()
            outputs = model(images).cpu().numpy()
            images_np = images.cpu().numpy()
            #loss_val = loss(outputs, true_coords)
            #print(loss_val)
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
                import numpy as np
                id_array = np.array(img_rgb)
                print("max pixel in image_depth is " , img_rgb)
                import cv2

                plot_image_with_annotations(img_rgb, img_depth, output, true_coord, f'{idx}.png')
                idx += 1

def calc_loss(model, device, test_loader):
    print("in calc_loss")
    avg_loss = 0 
    for i, data in enumerate(test_loader):

        inputs, labels = data['image'], data['coords'].float()
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        center_x_label = labels[:,0]
        center_y_label = labels[:,2]
        end_x_label = labels[:,1]
        end_y_label  = labels[:,3]
        radius_list = labels[:,4]

        #print(radius_list)

        labels = labels[:,:4]
        #print("shape is ",labels.shape)
        center_x_output = outputs[:,0]
        center_y_output = outputs[:,2]
        end_x_output = outputs[:,1]
        end_y_output  = outputs[:,3]

        distance_list = torch.sqrt((center_x_label - center_x_output)**2 + (center_y_label - center_y_output)**2)
        diff_list_dist = distance_list - radius_list
        diff_list_dist = torch.clamp(diff_list_dist, min=0)
        angle_list_outputs = torch.atan2((end_y_output-center_y_output),(end_x_output-center_x_output))*180/(torch.pi)
        angle_list_labels = torch.atan2((end_y_label-center_y_label),(end_x_label-center_x_label))*180/(torch.pi)
        diff_list_angle = torch.abs(angle_list_labels - angle_list_outputs)

        loss = 100*torch.mean(diff_list_dist) + 1020.4*(100/180)*torch.mean(diff_list_angle)
        avg_loss = avg_loss+ loss.item()
    print('loss is here ',avg_loss/(i+1))




# def test_model2(model, device, test_loader, img_name_list, img_root_dir):
#     from PIL import Image
#     idx = 0
#     with torch.no_grad():
#         pil_images = [Image.open(img_root_dir+"/" + image_path) for image_path in img_name_list]
#         transform = transforms.Compose([transforms.ToTensor()])
#         tensors = torch.stack([transform(image) for image in pil_images])
#         output = model(tensors) 

#         for data in test_loader:
#             images, true_coords = data['image'].to(device), data['coords'].cpu().numpy()
#             outputs = model(images).cpu().numpy()
#             images_np = images.cpu().numpy()
            
#             # for img, output, true_coord in zip(images_np, outputs, true_coords):
#             #     # Assuming the first three channels are RGB and fourth channel is depth
#             #     img_rgb = np.transpose(img[:3], (1, 2, 0))
#             #     img_depth = img[3]
#             #     plot_image_with_annotations(img_rgb, img_depth, output, true_coord, f'{idx}.png')
#             #     idx += 1

#             for img, output, true_coord in zip(images_np, outputs, true_coords):
#                 # Assuming the first channel is grayscale and the second is depth
#                 img_rgb = img[0]  # Grayscale
#                 img_depth = img[1]  # Depth
#                 plot_image_with_annotations(img_rgb, img_depth, output, true_coord, f'{idx}.png')
#                 idx += 1


def prepare_datasets(cfgs):
    unique_indices = range(814)
    unique_indices = range(561)
    #unique_indices = range(52)

    #train_indices, test_indices = train_test_split(unique_indices, test_size=0.3, random_state=42)
    train_indices, test_indices = train_test_split(unique_indices, test_size=0.3, random_state=42)
    #test_indices = [0]
    
    print("test_indices are ", test_indices)

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
    #model_path = cfgs["chkpnts_path"]
    test_loader = prepare_datasets(cfgs)
    model = load_model(model_path, device)
    test_model(model, device, test_loader)
    print("hey there")
    #calc_loss(model, device, test_loader)
