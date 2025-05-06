import os
import zipfile
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights

from dataset.spectrogram_dataset import SpectrogramDataset


# Basic ResNet18
class AudioNet(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )  # pretrained weights on ImageNet
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 2)  # binary classification
        self.model = model

    def forward(self, x):
        return self.model(x)


# Train
def train_one_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Train"):
        x = batch["spectrogram"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    print(f"Train Loss: {train_loss:.4f}")

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["spectrogram"].to(device)
            y = batch["label"].to(device)
            output = model(x)
            loss = criterion(output, y)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Val Split Loss: {val_loss:.4f}")


# Predict
def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test"):
            x = batch["spectrogram"].to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            preds.extend(pred.cpu().numpy())
    return preds


# Save to CSV
def save_submission_csv(preds, save_name):
    df = pd.DataFrame(preds)
    df.to_csv(save_name, index=False, header=False)


def main():
    # Instantiation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Train
    full_train_set = SpectrogramDataset("dataset/training_set")
    val_size = int(0.2 * len(full_train_set))
    train_size = len(full_train_set) - val_size
    train_set, val_split_set = random_split(full_train_set, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=32)
    val_split_loader = DataLoader(val_split_set, batch_size=32)
    train_one_epoch(model, train_loader, val_split_loader, criterion, optimizer, device)

    # Test
    val_set = SpectrogramDataset("dataset/validation_set")
    test_set = SpectrogramDataset("dataset/testing_set")
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)
    val_preds = predict(model, val_loader, device)
    test_preds = predict(model, test_loader, device)

    # Submission Process
    save_submission_csv(val_preds, "submissionA.csv")
    save_submission_csv(test_preds, "submissionB.csv")
    with zipfile.ZipFile("submission.zip", "w") as zipf:
        zipf.write("submissionA.csv")
        zipf.write("submissionB.csv")
    os.remove("submissionA.csv")
    os.remove("submissionB.csv")


if __name__ == "__main__":
    main()
