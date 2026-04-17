# This code displays the model building process for a classifier that
# can determine whether or not a photo object is recyclable.

# Imports
import os
import random
import shutil

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Mount Google Drive and set paths
from google.colab import drive
drive.mount('/content/drive')

base_dir         = "/content/drive/MyDrive/#3 - Data Science Project/RealWaste"
new_base_dir     = "/content/drive/MyDrive/#3 - Data Science Project/CleanedDataset"
resized_base_dir = "/content/drive/MyDrive/#3 - Data Science Project/ResizedDataset"
split_base       = "/content/drive/MyDrive/#3 - Data Science Project/SplitDataset"

# Step 1: Group images into Recyclable / Non-Recyclable
# Remove the "Plastic" category and bin remaining classes by recyclability

recyclable_classes     = ["Cardboard", "Paper", "Glass", "Metal"]
non_recyclable_classes = ["Miscellaneous Trash", "Textile Trash", "Food Organics", "Vegetation"]

recyclable_dir     = os.path.join(new_base_dir, "Recyclable")
non_recyclable_dir = os.path.join(new_base_dir, "Non_Recyclable")

os.makedirs(recyclable_dir,     exist_ok=True)
os.makedirs(non_recyclable_dir, exist_ok=True)

def copy_images(class_list, target_dir):
    for class_name in class_list:
        class_path = os.path.join(base_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Skipping missing folder: {class_name}")
            continue
        for img_name in os.listdir(class_path):
            src = os.path.join(class_path, img_name)
            # Prefix filename with class name to avoid collisions
            dst = os.path.join(target_dir, f"{class_name}_{img_name}")
            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(f"Error copying {src}: {e}")

copy_images(recyclable_classes,     recyclable_dir)
copy_images(non_recyclable_classes, non_recyclable_dir)

print("Recyclable count:",     len(os.listdir(recyclable_dir)))
print("Non-Recyclable count:", len(os.listdir(non_recyclable_dir)))

# Step 2: Class distribution chart
labels = ["Recyclable", "Non-Recyclable"]
counts = [len(os.listdir(recyclable_dir)), len(os.listdir(non_recyclable_dir))]

plt.figure()
plt.bar(labels, counts)
plt.title("Class Distribution (Cleaned Dataset)")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.show()

# Step 3: Show sample images from each class
def show_samples(folder, title):
    images = random.sample(os.listdir(folder), 6)
    plt.figure(figsize=(8, 6))
    for i, img_name in enumerate(images):
        img = Image.open(os.path.join(folder, img_name))
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.show()

show_samples(recyclable_dir,     "Recyclable")
show_samples(non_recyclable_dir, "Non-Recyclable")

# Step 4: Resize all images to 224x224
resized_recyclable     = os.path.join(resized_base_dir, "Recyclable")
resized_non_recyclable = os.path.join(resized_base_dir, "Non_Recyclable")

os.makedirs(resized_recyclable,     exist_ok=True)
os.makedirs(resized_non_recyclable, exist_ok=True)

def resize_images(src_dir, dst_dir):
    for img_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)
        try:
            img = Image.open(src_path).convert("RGB")
            img = img.resize((224, 224))
            img.save(dst_path)
        except Exception as e:
            print(f"Skipping {img_name}: {e}")

resize_images(recyclable_dir,     resized_recyclable)
resize_images(non_recyclable_dir, resized_non_recyclable)

print("Resizing complete!")

# Step 5: Split into train / val / test (80 / 10 / 10)
train_dir = os.path.join(split_base, "train")
val_dir   = os.path.join(split_base, "val")
test_dir  = os.path.join(split_base, "test")

classes = ["Recyclable", "Non_Recyclable"]

for split in [train_dir, val_dir, test_dir]:
    for cls in classes:
        os.makedirs(os.path.join(split, cls), exist_ok=True)

def split_data(src_dir, train_dst, val_dst, test_dst):
    images = os.listdir(src_dir)
    random.shuffle(images)

    total     = len(images)
    train_end = int(0.8 * total)
    val_end   = int(0.9 * total)

    for imgs, dst in [
        (images[:train_end],        train_dst),
        (images[train_end:val_end], val_dst),
        (images[val_end:],          test_dst),
    ]:
        for img in imgs:
            shutil.copy(os.path.join(src_dir, img), os.path.join(dst, img))

split_data(resized_recyclable,
           os.path.join(train_dir, "Recyclable"),
           os.path.join(val_dir,   "Recyclable"),
           os.path.join(test_dir,  "Recyclable"))

split_data(resized_non_recyclable,
           os.path.join(train_dir, "Non_Recyclable"),
           os.path.join(val_dir,   "Non_Recyclable"),
           os.path.join(test_dir,  "Non_Recyclable"))

print("Split complete!")

# Step 6: Create DataLoaders
# Training uses augmentation; val/test use only normalization
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data   = datasets.ImageFolder(val_dir,   transform=val_test_transform)
test_data  = datasets.ImageFolder(test_dir,  transform=val_test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32)
test_loader  = DataLoader(test_data,  batch_size=32)

print("Data loaded!")

# Step 7: Load pretrained MobileNetV2 and replace the classifier head for binary output
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(device)

# Step 8: Train with early stopping
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses  = []
val_losses    = []
best_val_loss = float('inf')
patience      = 3
counter       = 0

for epoch in range(10):
    # Training pass
    model.train()
    running_train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation pass
    model.eval()
    running_val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            running_val_loss += criterion(outputs, labels).item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Save best model and check early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# Step 9: Plot training vs validation loss
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# Step 10: Evaluate on test set
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(device)
        outputs = model(images)
        preds   = torch.argmax(outputs, dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=["Recyclable", "Non-Recyclable"]))

# Step 11: Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Recyclable", "Non-Recyclable"],
            yticklabels=["Recyclable", "Non-Recyclable"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 12: Metrics bar chart
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)

plt.figure()
plt.bar(["Precision", "Recall", "F1"], [precision, recall, f1])
plt.title("Model Performance Metrics")
plt.ylim(0, 1)
plt.show()
