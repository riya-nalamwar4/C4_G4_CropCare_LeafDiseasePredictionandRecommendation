import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

DATA_DIR = "dataset"
MODEL_DIR = "model"
BATCH_SIZE = 16
EPOCHS = 15
LR = 0.0001
NUM_CLASSES = 6

os.makedirs(MODEL_DIR, exist_ok=True)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print("Classes:", train_dataset.classes)

# ---------------- MODEL ----------------
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_acc = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()
    train_loss = 0
    train_correct = 0

    for img, labels in tqdm(train_loader):
        img, labels = img.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss = 0
    val_correct = 0
    model.eval()

    with torch.no_grad():
        for img, labels in val_loader:
            img, labels = img.to(device), labels.to(device)
            outputs = model(img)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_dataset)
    val_acc   = val_correct / len(val_dataset)

    print(f"Train Acc: {train_acc:.4f}   Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "model/ResNet50_6class.pth")
        print("✔ Saved best model")

print("Training Completed!")
