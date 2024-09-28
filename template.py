import os
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import wandb
from pytorch_lightning.loggers import WandbLogger


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        """
        INSERT MODEL ARCHITECTURE HERE
        """
    
    def forward(self, x):
        """
        INSERT FORWARD PASS HERE
        """
        pass
    
    def training_step(self, batch, batch_idx):
        """
        INSERT TRAINING STEP HERE
        """
        self.log('train_accuracy', train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        INSERT VALIDATION STEP HERE
        """
        self.log('val_accuracy', val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        """
        INSERT OPTIMIZER HERE
        """

# Data preparation
dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, batch_size=32)

val_dataset = MNIST('', train=False, download=True, transform=transforms.ToTensor())
val_loader = DataLoader(val_dataset, batch_size=32)


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
    trainer.fit(model, train_loader, val_loader)

    val_accuracy = trainer.callback_metrics['val_accuracy'].item()                                                

    
    os.makedirs(args.out_dir, exist_ok=True)                                                                      
    with open(os.path.join(args.out_dir, 'val_accuracy.txt'), 'w') as f:                                          
        f.write(f"{val_accuracy}")  

    wandb.finish()