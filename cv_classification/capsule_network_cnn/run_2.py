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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, routing_iters=3):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_capsules = num_capsules
        self.routing_iters = routing_iters

        self.W = nn.Parameter(torch.randn(1, num_route_nodes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        b_ij = torch.zeros(1, self.num_route_nodes, self.num_capsules, 1).to(x.device)

        for iteration in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            if iteration < self.routing_iters - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), v_j)
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    @staticmethod
    def squash(s_j):
        s_j_norm = torch.norm(s_j, dim=-1, keepdim=True)
        return (s_j_norm**2 / (1 + s_j_norm**2)) * (s_j / s_j_norm)

class Model(LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=6*6*256, in_channels=256, out_channels=8)
        self.digit_capsules = CapsuleLayer(num_capsules=10, num_route_nodes=6*6*8, in_channels=8, out_channels=16)
        self.decoder = nn.Sequential(
            nn.Linear(16*10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32*32*3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        num_route_nodes = x.size(2) * x.size(3)
        x = self.primary_capsules(x.view(x.size(0), num_route_nodes, 256))
        num_route_nodes = x.size(1) * x.size(2)
        x = self.digit_capsules(x.view(x.size(0), num_route_nodes, 8)).squeeze().transpose(0, 1)
        y = torch.eye(10).to(x.device).index_select(dim=0, index=torch.argmax(x, dim=1))
        reconstructions = self.decoder((x * y).view(x.size(0), -1))
        return x, reconstructions

    def training_step(self, batch, batch_idx):
        data, target = batch
        output, reconstructions = self(data)
        loss = self.margin_loss(output, target) + self.reconstruction_loss(reconstructions, data)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output, reconstructions = self(data)
        loss = self.margin_loss(output, target) + self.reconstruction_loss(reconstructions, data)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def margin_loss(self, output, target):
        batch_size = target.size(0)
        v_c = torch.sqrt((output**2).sum(dim=2, keepdim=True))
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        loss = target * left + 0.5 * (1.0 - target) * right
        return loss.sum(dim=1).mean()

    def reconstruction_loss(self, reconstructions, data):
        return F.mse_loss(reconstructions, data)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
        return DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        val_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)
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
        max_epochs=3,
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

    val_accuracy = trainer.callback_metrics['val_accuracy'].item()                                                

    
    os.makedirs(args.out_dir, exist_ok=True)                                                                      
    with open(os.path.join(args.out_dir, 'val_accuracy.txt'), 'w') as f:                                          
        f.write(f"{val_accuracy}")  

    wandb.finish()
