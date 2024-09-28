import os
import torchvision
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from capsule_layers import CapsuleLayer, PrimaryCapsules
import wandb
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = PrimaryCapsules(num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=10, num_route_nodes=32 * 6 * 6, in_channels=8, out_channels=16)
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32 * 32 * 3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        x = x.norm(dim=-1)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.eye(10, device=self.device).index_select(dim=0, index=y)
        y_pred = self(x)
        loss = self.margin_loss(y, y_pred)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.eye(10, device=self.device).index_select(dim=0, index=y)
        y_pred = self(x)
        loss = self.margin_loss(y, y_pred)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
        
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
        return DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        val_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)
        return DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

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
        max_epochs=3,
        accelerator='auto',
        logger=wandb_logger
    )

    # Train the model
    trainer.fit(model)

    val_loss = trainer.callback_metrics['val_loss'].item()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'val_loss.txt'), 'w') as f:
        f.write(f"{val_loss}")

    wandb.finish()
