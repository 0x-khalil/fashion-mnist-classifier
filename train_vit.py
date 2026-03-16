import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vit_model import FashionMNISTViT


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),                 # ViT input size
    transforms.Grayscale(num_output_channels=3),   # convert 1 channel -> 3
    transforms.ToTensor(),
])


# Dataset
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# Model
model = FashionMNISTViT().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)


# Training function
def train():
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print("Training Loss:", running_loss / len(train_loader))


# Evaluation function
def evaluate():
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("Test Accuracy:", accuracy)


# Training loop
epochs = 5

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train()
    evaluate()


# Save model
torch.save(model.state_dict(), "vit_fashionmnist.pth")

print("Model saved!")
