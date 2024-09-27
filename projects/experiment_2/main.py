import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import wandb
from dataclasses import dataclass

# Define hyperparameters and configuration
@dataclass
class Config:
    learning_rate: float = 0.005
    batch_size: int = 64
    epochs: int = 2
    architecture_description: str = "Conv(1, 32, 3x3).ReLU().MaxPool(2x2) -> Conv(32, 64, 3x3).ReLU().MaxPool(2x2) -> FC(64*7*7, 128).ReLU() -> FC(128, 10)"
    run_name: str = "idea1/run_3"
    project_name: str = "ai-engineer"
    group_name: str = "mnist_classification"

config = Config()

# Initialize Weights and Biases
wandb.init(
    name=config.run_name,
    project=config.project_name,
    config={
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "architecture_description": config.architecture_description,
    },
    group=config.group_name
)

# Define the neural network architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = MNIST(root='.', train=True, download=True, transform=transform)
mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])
train_loader = DataLoader(mnist_train, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=config.batch_size)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Training loop
for epoch in range(config.epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        train_total += target.size(0)
        train_correct += predicted.eq(target).sum().item()

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            _, predicted = output.max(1)
            val_total += target.size(0)
            val_correct += predicted.eq(target).sum().item()

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

    print(f"Epoch {epoch+1}/{config.epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Finish the wandb run
wandb.finish()