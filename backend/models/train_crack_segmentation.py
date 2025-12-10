import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- Minimal UNet implementation ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.seq(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)
    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        bn = self.bottleneck(p3)
        u3 = self.up3(bn)
        concat3 = torch.cat([u3, d3], dim=1)
        conv3 = self.conv3(concat3)
        u2 = self.up2(conv3)
        concat2 = torch.cat([u2, d2], dim=1)
        conv2 = self.conv2(concat2)
        u1 = self.up1(conv2)
        concat1 = torch.cat([u1, d1], dim=1)
        conv1 = self.conv1(concat1)
        out = self.out_conv(conv1)
        return out

# --- Dataset for paired images and masks ---
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        # Image transform
        if self.transform:
            image = self.transform(image)
        # Mask transform (resize and tensorize)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()   # Binarize
        return image, mask


# --- Paths ---
train_img = r'C:\Users\PAVAN MANIKANTA\Downloads\MineSafety\backend\data\crack_segmentation_split\train\images'
train_mask = r'C:\Users\PAVAN MANIKANTA\Downloads\MineSafety\backend\data\crack_segmentation_split\train\masks'
val_img = r'C:\Users\PAVAN MANIKANTA\Downloads\MineSafety\backend\data\crack_segmentation_split\val\images'
val_mask = r'C:\Users\PAVAN MANIKANTA\Downloads\MineSafety\backend\data\crack_segmentation_split\val\masks'
save_path = r'C:\Users\PAVAN MANIKANTA\Downloads\MineSafety\backend\models\saved_models\crack_segmenter.pth'

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_ds = SegmentationDataset(train_img, train_mask, img_transform)
val_ds = SegmentationDataset(val_img, val_mask, img_transform)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
print(f"Device: {device}\n")

for epoch in range(10):
    model.train()
    total_loss = 0
    print(f"\nEpoch {epoch+1} ----------------------")
    for batch_idx, (imgs, masks) in enumerate(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 5 == 0 or batch_idx == len(train_loader)-1:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}, Batch Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(val_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            if batch_idx == len(val_loader)-1:
                print(f"  Validation: last batch loss: {loss.item():.4f}")

    print(f"Val Loss: {val_loss / len(val_loader):.4f}")

os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Segmentation model saved to {save_path}")
