import torch.optim as optim
from modified_unet import ModifiedUNet
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split

class RGBDDataset(Dataset):
    def __init__(self, csv_file, rgb_root_dir, depth_root_dir, transform=None, indices=None):
        self.annotations = pd.read_csv(csv_file)
        if indices is not None:
            self.annotations = self.annotations.iloc[indices]
        self.rgb_root_dir = rgb_root_dir
        self.depth_root_dir = depth_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        rgb_img_name = os.path.join(self.rgb_root_dir, self.annotations.iloc[idx, 0])
        depth_img_name = os.path.join(self.depth_root_dir, self.annotations.iloc[idx, 0].replace('RGB', 'Depth'))

        try:
            rgb_image = Image.open(rgb_img_name).convert('RGB')
            depth_image = Image.open(depth_img_name).convert('L')

            # stack RGB and depth into a single ndarray
            image = np.concatenate((np.array(rgb_image), np.expand_dims(np.array(depth_image), axis=2)), axis=2)

            if self.transform:
                image = self.transform(image)

            coords = self.annotations.iloc[idx, 1:5].values
            coords = np.array(coords).astype(np.float32)
            sample = {'image': image, 'coords': coords}

            return sample
        

        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            raise


full_annotations = pd.read_csv('/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/processed_data/0-30/annotations/annotation_T0-T9.csv')

indices = range(len(full_annotations))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = RGBDDataset(
    csv_file='/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/processed_data/0-30/annotations/annotation_T0-T9.csv',
    rgb_root_dir='/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/processed_data/0-30/cropped_rgb/T0-T9',
    depth_root_dir='/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/processed_data/0-30/per_dataset_normalized_depth/T0-T9',
    transform=transforms.Compose([transforms.ToTensor()]),
    indices=train_indices
)

test_dataset = RGBDDataset(
    csv_file='/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/processed_data/0-30/annotations/annotation_T0-T9.csv',
    rgb_root_dir='/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/processed_data/0-30/cropped_rgb/T0-T9',
    depth_root_dir='/Users/YifeiHu/Desktop/Cornell/Emprise Lab/Shirt_Perception/processed_data/0-30/per_dataset_normalized_depth/T0-T9',
    transform=transforms.Compose([transforms.ToTensor()]),
    indices=test_indices
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

model = ModifiedUNet()
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f'Epoch {epoch}')

        for i, data in enumerate(dataloader):
            inputs, labels = data['image'], data['coords']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9: 
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        if epoch % 5 == 4:  # Save every 5 epochs
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

train_model(model, train_loader, criterion, optimizer, num_epochs=25)


def test_model(model, dataloader, criterion):
    model.eval() 
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data['image'], data['coords']
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    print(f'Test Loss: {avg_loss:.4f}')
    return avg_loss


model = ModifiedUNet()
model_path = 'model_epoch_X.pth' 
model.load_state_dict(torch.load(model_path))
# model.cuda()

criterion = torch.nn.MSELoss()

test_loss = test_model(model, test_loader, criterion)

