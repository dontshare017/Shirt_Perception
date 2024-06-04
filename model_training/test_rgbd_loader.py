import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rgbd_datase import RGBDDataset

def plot_image_with_annotations(image, coords):
    fig, ax = plt.subplots(1)
    ax.imshow(image[:, :, :3])  # display only the RGB channels

    x_start, x_end, y_start, y_end = coords
    width = x_end - x_start
    height = y_end - y_start

    rect = patches.Rectangle((x_start, y_start), width, height, linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    ax.plot([x_start, x_end], [y_start, y_end], 'go')
    ax.plot([x_start, x_end], [y_start, y_end], 'g--')
    plt.show()

def test_dataset(dataset, num_samples=5):
    print(f"Testing dataset with {len(dataset)} samples")
    if len(dataset) == 0:
        print("No data to display")
        return

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample['image']
        coords = sample['coords']
        print(f"Sample {i}: Image shape {image.shape}, Coords: {coords}")

        if i < 3: 
            plot_image_with_annotations(image, coords)

dataset = RGBDDataset(root_rgb_dir="/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/downsampled_final/rgb", root_depth_dir="/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/downsampled_final/depth")
test_dataset(dataset)
print(len(dataset.data))
