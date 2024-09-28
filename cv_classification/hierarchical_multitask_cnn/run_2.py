import os
import torchvision
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
import wandb
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.validation_step_outputs = []
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc_super = nn.Linear(256 * 4 * 4, 20)  # Superclass head
        self.fc_sub = nn.Linear(256 * 4 * 4, 100)  # Subclass head

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        super_logits = self.fc_super(x)
        sub_logits = self.fc_sub(x)
        return super_logits, sub_logits

    def training_step(self, batch, batch_idx):
        images, sub_labels = batch
        super_labels = self.map_sub_to_super(sub_labels)
        super_logits, sub_logits = self(images)
        loss_super = F.cross_entropy(super_logits, super_labels)
        loss_sub = F.cross_entropy(sub_logits, sub_labels)
        loss = 0.5 * loss_super + 0.5 * loss_sub  # Adjust alpha and beta as needed
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, sub_labels = batch
        super_labels = self.map_sub_to_super(sub_labels)
        super_logits, sub_logits = self(images)
        loss_super = F.cross_entropy(super_logits, super_labels)
        loss_sub = F.cross_entropy(sub_logits, sub_labels)
        loss = 0.5 * loss_super + 0.5 * loss_sub  # Adjust alpha and beta as needed

        # Calculate accuracy
        super_preds = torch.argmax(super_logits, dim=1)
        sub_preds = torch.argmax(sub_logits, dim=1)
        super_acc = (super_preds == super_labels).float().mean()
        sub_acc = (sub_preds == sub_labels).float().mean()

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_super_acc', super_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_sub_acc', sub_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        output = {'val_loss': loss, 'val_super_acc': super_acc, 'val_sub_acc': sub_acc}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        avg_super_acc = torch.stack([x['val_super_acc'] for x in self.validation_step_outputs]).mean()
        avg_sub_acc = torch.stack([x['val_sub_acc'] for x in self.validation_step_outputs]).mean()
        self.log('val_super_acc_epoch', avg_super_acc, prog_bar=True, logger=True)
        self.log('val_sub_acc_epoch', avg_sub_acc, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()  # Clear the outputs after logging
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
        
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        return DataLoader(train_dataset, batch_size=64, shuffle=True)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        return DataLoader(val_dataset, batch_size=64, shuffle=False)

    def map_sub_to_super(self, sub_labels):
        # Mapping from subclass to superclass
        sub_to_super = {
            0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3,
            10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11,
            20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15,
            30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5,
            40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10,
            50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17,
            60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19,
            70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13,
            80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 8, 88: 19, 89: 18,
            90: 1, 91: 2, 92: 10, 93: 0, 94: 1, 95: 16, 96: 12, 97: 9, 98: 13, 99: 15
        }
        return torch.tensor([sub_to_super[label.item()] for label in sub_labels], device=sub_labels.device)

def main():
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

    val_super_accuracy = trainer.callback_metrics['val_super_acc_epoch'].item()
    val_sub_accuracy = trainer.callback_metrics['val_sub_acc_epoch'].item()

    
    os.makedirs(args.out_dir, exist_ok=True)                                                                      
    with open(os.path.join(args.out_dir, 'val_accuracy.txt'), 'w') as f:
        f.write(f"Superclass Accuracy: {val_super_accuracy}\n")
        f.write(f"Subclass Accuracy: {val_sub_accuracy}\n")

    wandb.finish()

if __name__ == "__main__":
    main()
