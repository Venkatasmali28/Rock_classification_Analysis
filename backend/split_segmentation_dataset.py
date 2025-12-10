# split_segmentation_dataset.py

import os
import shutil
import random

def split_segmentation_dataset(src_img_dir, src_mask_dir, dst_base_dir, train_ratio=0.8):
    train_img_dir = os.path.join(dst_base_dir, 'train', 'images')
    train_mask_dir = os.path.join(dst_base_dir, 'train', 'masks')
    val_img_dir = os.path.join(dst_base_dir, 'val', 'images')
    val_mask_dir = os.path.join(dst_base_dir, 'val', 'masks')

    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        os.makedirs(d, exist_ok=True)

    images = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Copy training images and masks
    for img in train_images:
        src_img_path = os.path.join(src_img_dir, img)
        src_mask_path = os.path.join(src_mask_dir, img)
        shutil.copy(src_img_path, os.path.join(train_img_dir, img))
        shutil.copy(src_mask_path, os.path.join(train_mask_dir, img))

    # Copy validation images and masks
    for img in val_images:
        src_img_path = os.path.join(src_img_dir, img)
        src_mask_path = os.path.join(src_mask_dir, img)
        shutil.copy(src_img_path, os.path.join(val_img_dir, img))
        shutil.copy(src_mask_path, os.path.join(val_mask_dir, img))

    print(f"Done! Train: {len(train_images)} images, Val: {len(val_images)} images.")

if __name__ == "__main__":
    # Example usage:
  split_segmentation_dataset(
    src_img_dir=r'C:\Users\PAVAN MANIKANTA\Downloads\MineSafety\backend\data\crack_segmentation\images',
    src_mask_dir=r'C:\Users\PAVAN MANIKANTA\Downloads\MineSafety\backend\data\crack_segmentation\masks',
    dst_base_dir=r'C:\Users\PAVAN MANIKANTA\Downloads\MineSafety\backend\data\crack_segmentation_split',
    train_ratio=0.8
)

