import os
import torchvision
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import wandb
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 100)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        train_accuracy = torch.sum(preds == y).item() / (len(y) * 1.0)
        self.log('train_accuracy', train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        val_accuracy = torch.sum(preds == y).item() / (len(y) * 1.0)
        self.log('val_accuracy', val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)
        
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        return DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        return DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, required=True)
    parser.add_argument('--wandb_name', type=str, required=True)
    parser.add_argument('--wandb_group', type=str, required=True)
    args = parser.parse_args()

    # Initialize wandb logger
    wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_name, group=args.wandb_group)

    # Create the model and trainer
    model = Model()

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        logger=wandb_logger
    )

    # Train the model
    trainer.fit(model)

    val_accuracy = trainer.callback_metrics['val_accuracy'].item()                                                

    
    os.makedirs(args.out_dir, exist_ok=True)                                                                      
    with open(os.path.join(args.out_dir, 'val_accuracy.txt'), 'w') as f:                                          
        f.write(f"{val_accuracy}")  

    wandb.finish()
