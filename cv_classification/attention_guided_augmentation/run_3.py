import os
import torchvision
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cbam import CBAM
from torchvision import transforms
from torchvision.datasets import MNIST
import wandb
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=False, num_classes=100)
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
    
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x, _, _ = self.cbam1(x)
        
        x = self.model.layer2(x)
        x, _, _ = self.cbam2(x)
        
        x = self.model.layer3(x)
        x, _, _ = self.cbam3(x)
        
        x = self.model.layer4(x)
        x, _, _ = self.cbam4(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
        
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        return DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
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
