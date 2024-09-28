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

def target_transform(target):
    super_class = target // 5
    return torch.tensor([super_class, target])

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_shared = nn.Linear(256 * 4 * 4, 512)
        
        # Superclass head
        self.fc_super = nn.Linear(512, 20)
        
        # Subclass head
        self.fc_sub = nn.Linear(512, 100)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc_shared(x))
        
        super_logits = self.fc_super(x)
        sub_logits = self.fc_sub(x)
        
        return super_logits, sub_logits
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        super_labels, sub_labels = labels[:, 0].long(), labels[:, 1].long()
        super_logits, sub_logits = self(images)
        
        loss_super = F.cross_entropy(super_logits, super_labels)
        loss_sub = F.cross_entropy(sub_logits, sub_labels)
        
        alpha = 0.5
        beta = 0.5
        loss = alpha * loss_super + beta * loss_sub
        
        super_preds = torch.argmax(super_logits, dim=1)
        sub_preds = torch.argmax(sub_logits, dim=1)
        
        train_super_accuracy = torch.sum(super_preds == super_labels).item() / len(super_labels)
        train_sub_accuracy = torch.sum(sub_preds == sub_labels).item() / len(sub_labels)
        
        self.log('train_super_accuracy', train_super_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_sub_accuracy', train_sub_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        super_labels, sub_labels = labels[:, 0].long(), labels[:, 1].long()
        super_logits, sub_logits = self(images)
        
        loss_super = F.cross_entropy(super_logits, super_labels)
        loss_sub = F.cross_entropy(sub_logits, sub_labels)
        
        alpha = 0.5
        beta = 0.5
        loss = alpha * loss_super + beta * loss_sub
        
        super_preds = torch.argmax(super_logits, dim=1)
        sub_preds = torch.argmax(sub_logits, dim=1)
        
        val_super_accuracy = torch.sum(super_preds == super_labels).item() / len(super_labels)
        val_sub_accuracy = torch.sum(sub_preds == sub_labels).item() / len(sub_labels)
        
        self.log('val_super_accuracy', val_super_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_sub_accuracy', val_sub_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        return optimizer
        
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        return trainloader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform, target_transform=target_transform)
        valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
        return valloader

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
