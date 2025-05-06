import os
import zipfile
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet34, ResNet34_Weights

from dataset.spectrogram_dataset import SpectrogramDataset


# Enhanced ResNet-34
class AudioNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(
                base.conv1.weight.mean(dim=1, keepdim=True)
            )
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    full_train_set = SpectrogramDataset("dataset/training_set")
    val_size = int(0.2 * len(full_train_set))
    train_size = len(full_train_set) - val_size
    train_set, val_split_set = random_split(full_train_set, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_split_loader = DataLoader(val_split_set, batch_size=32)

    for _ in range(2):
        train_one_epoch(
            model, train_loader, val_split_loader, criterion, optimizer, device
        )

    val_set = SpectrogramDataset("dataset/validation_set")
    test_set = SpectrogramDataset("dataset/testing_set")
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    val_preds = predict(model, val_loader, device)
    test_preds = predict(model, test_loader, device)

    save_submission_csv(val_preds, "submissionA.csv")
    save_submission_csv(test_preds, "submissionB.csv")
    with zipfile.ZipFile("submission.zip", "w") as zipf:
        zipf.write("submissionA.csv")
        zipf.write("submissionB.csv")
    os.remove("submissionA.csv")
    os.remove("submissionB.csv")


if __name__ == "__main__":
    main()
