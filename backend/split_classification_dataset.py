import os
import shutil
import random

def get_all_class_subfolders(src_dir):
    class_subfolders = []
    # One level deep only (e.g., Igneous/Basalt)
    for cls in os.listdir(src_dir):
        cls_path = os.path.join(src_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for subcls in os.listdir(cls_path):
            subcls_path = os.path.join(cls_path, subcls)
            if os.path.isdir(subcls_path):
                class_subfolders.append((cls, subcls, subcls_path))
    return class_subfolders

def split_dataset(
    src_dir,
    dst_dir,
    train_ratio=0.8
):
    os.makedirs(dst_dir, exist_ok=True)
    train_dir = os.path.join(dst_dir, "train")
    val_dir = os.path.join(dst_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for cls, subcls, subcls_path in get_all_class_subfolders(src_dir):
        images = [img for img in os.listdir(subcls_path) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Class name will be combined, e.g., Igneous_Basalt
        class_name = f"{cls}_{subcls}"
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        for img in train_imgs:
            src = os.path.join(subcls_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy(src, dst)

        for img in val_imgs:
            src = os.path.join(subcls_path, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copy(src, dst)

        print(f"Class: {class_name} -> Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    print("Done! Split complete.")

if __name__ == '__main__':
    split_dataset(
        src_dir=r'C:\Users\PAVAN MANIKANTA\Downloads\MineSafety\backend\data\rock_classification',
        dst_dir=r'C:\Users\PAVAN MANIKANTA\Downloads\MineSafety\backend\data\rock_classification_split',
        train_ratio=0.8
    )
