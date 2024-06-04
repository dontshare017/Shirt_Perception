import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
from datetime import datetime
from tensorboardX import SummaryWriter
from modified_unet import ModifiedUNet
from rgbd_dataset_split import RGBDDataset
import pandas as pd
from sklearn.model_selection import train_test_split

with open('config_split.json', 'r') as cfg_file:
    cfgs = json.load(cfg_file)

class BaseTrain:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.init_dirs()
        self.init_datasets()
        self.init_model()
        self.init_tboard()
        self.load_model()
        self.last_epoch = 0

    def init_dirs(self):
        runspath = self.cfgs["runspath"]
        try:
            self.run_id = str(max([int(run_id) for run_id in os.listdir(runspath) if run_id.isdecimal()]) + 1)
        except ValueError:
            self.run_id = '1'

        self.model_path = os.path.join(runspath, self.run_id)
        os.makedirs(self.model_path, exist_ok=True)
        with open(os.path.join(self.model_path, 'cfgs.json'), 'w') as f:
            json.dump(self.cfgs, f, indent=4)
        self.chkpnts_path = os.path.join(self.model_path, "chkpnts")
        os.makedirs(self.chkpnts_path, exist_ok=True)

    def init_datasets(self):
        # Assuming the total number of unique data points is 1000 (excluding color augmented data)
        unique_indices = range(1000)
        train_indices, test_indices = train_test_split(unique_indices, test_size=0.2, random_state=42)

        # include indices for color augmented data in training indices
        train_indices = list(train_indices) + list(range(1000, 1975))

        transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = RGBDDataset(
            root_rgb_dir=self.cfgs["rgb_root_dir"],
            root_depth_dir=self.cfgs["depth_root_dir"],
            root_aug_rgb_dir=self.cfgs["aug_rgb_root_dir"],
            transform=transform,
            indices=train_indices
        )
        self.test_dataset = RGBDDataset(
            root_rgb_dir=self.cfgs["rgb_root_dir"],
            root_depth_dir=self.cfgs["depth_root_dir"],
            root_aug_rgb_dir=self.cfgs["aug_rgb_root_dir"],
            transform=transform,
            indices=test_indices
        )
        print(len(self.train_dataset.data))
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfgs["batch_size"], shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.cfgs["batch_size"], shuffle=False, num_workers=4)

    def init_model(self):
        self.model = ModifiedUNet()
        if torch.cuda.is_available():
            self.model.cuda()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfgs["lr"])
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.cfgs["step_size"], gamma=self.cfgs["gamma"])
    
    def load_model(self):
        if self.cfgs["chkpnts_path"] != "":
            checkpoint = torch.load(self.cfgs["chkpnts_path"])
            print("Checkpoint keys:", checkpoint.keys()) 
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.last_epoch = checkpoint.get('epoch', 0)
            

    def init_tboard(self):
        log_dir = os.path.join(self.model_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def train(self):
        for epoch in range(self.cfgs["epochs"]):
            self.model.train()
            running_loss = 0.0
            epoch_loss = 0.0
            picture_count = 0
            for i, data in enumerate(self.train_loader):
                inputs, labels = data['image'], data['coords'].float()
                inputs, labels = inputs.cuda(), labels.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:  # every 10 batches
                    avg_loss = running_loss / 10
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {avg_loss:.4f}')
                    # self.writer.add_scalar('Training Loss', avg_loss, epoch * len(self.train_loader) + i)
                    epoch_loss += running_loss
                    picture_count += 10 * self.cfgs["batch_size"]
                    running_loss = 0.0

            pic_avg_loss = epoch_loss / picture_count
            self.writer.add_scalar('Training Loss', pic_avg_loss, epoch)
                

            self.scheduler.step()
            # save every 2 epochs and include timestamp
            checkpoint = {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict()
            }
            checkpoint_filename = f'model_epoch_{epoch}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            try:
                if epoch % 2 == 0:
                    torch.save(checkpoint, os.path.join(self.chkpnts_path, checkpoint_filename))
                    print(f"Checkpoint saved: {checkpoint_filename}")
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")


        self.writer.close()

if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = BaseTrain(cfgs)
    t.train()
