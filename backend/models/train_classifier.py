import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# Paths
train_dir = r'C:\Users\PAVAN MANIKANTA\Downloads\MineSafety\backend\data\rock_classification_split\train'
val_dir = r'C:\Users\PAVAN MANIKANTA\Downloads\MineSafety\backend\data\rock_classification_split\val'
save_path = r'C:\Users\PAVAN MANIKANTA\Downloads\MineSafety\backend\models\saved_models\rock_classifier.pth'


# Data pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

batch_size = 32
train_ds = datasets.ImageFolder(train_dir, transform=transform)
val_ds = datasets.ImageFolder(val_dir, transform=transform)
num_classes = len(train_ds.classes)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

# Model and setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    val_acc = correct / len(val_ds)
    print(f"Val Accuracy: {val_acc:.4f}")

torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
