# This script provides some basic exploratory data analysis for the RealWaste dataset, as well as for the cleaned dataset that classifies the categories of waste into recyclable and non-recyclable.
"""

#class sizes
#example images
from google.colab import drive
drive.mount('/content/drive') # this code utilizes Google Drive; don't have to do this

data_dir = "/content/drive/MyDrive/#3 - Data Science Project/RealWaste" # edit as needed

import os

print(os.listdir(data_dir))

class_counts = {}

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)

    if os.path.isdir(class_path):  # make sure it's a folder
        count = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        class_counts[class_name] = count

print(class_counts)

import pandas as pd

df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
df = df.sort_values(by="Count", ascending=False)

print(df)

import matplotlib.pyplot as plt

# Class Distribution plot

plt.figure()
plt.bar(df["Class"], df["Count"])
plt.xticks(rotation=45)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.show()

# Image count
total_images = sum(class_counts.values())
print("Total images:", total_images)

# Largest and smallest classes

print("Largest class:", df.iloc[0])
print("Smallest class:", df.iloc[-1])

# Creating binary category of recyclable and non-recyclable

import os
import shutil

base_dir = "/content/drive/MyDrive/#3 - Data Science Project/RealWaste" # edit as needed
new_base_dir = "/content/drive/MyDrive/#3 - Data Science Project/CleanedDataset" # edit as needed

recyclable_dir = os.path.join(new_base_dir, "Recyclable")
non_recyclable_dir = os.path.join(new_base_dir, "Non_Recyclable")

os.makedirs(recyclable_dir, exist_ok=True)
os.makedirs(non_recyclable_dir, exist_ok=True)

recyclable_classes = ["Cardboard", "Paper", "Glass", "Metal"]
non_recyclable_classes = ["Miscellaneous Trash", "Textile Trash", "Food Organics", "Vegetation"]

# Copying images to new folders

def copy_images(class_list, target_dir):
    for class_name in class_list:
        class_path = os.path.join(base_dir, class_name)

        if not os.path.exists(class_path):
            print(f"Skipping missing folder: {class_name}")
            continue

        for img_name in os.listdir(class_path):
            src = os.path.join(class_path, img_name)

            # Add class name to avoid filename collisions
            dst = os.path.join(target_dir, f"{class_name}_{img_name}")

            try:
                shutil.copy(src, dst)
            except Exception as e:
                print(f"Error copying {src}: {e}")

copy_images(recyclable_classes, recyclable_dir)
copy_images(non_recyclable_classes, non_recyclable_dir)

print("Recyclable count:", len(os.listdir(recyclable_dir)))
print("Non-Recyclable count:", len(os.listdir(non_recyclable_dir)))

# Cleaned Dataset counts

import matplotlib.pyplot as plt

labels = ["Recyclable", "Non-Recyclable"]
counts = [
    len(os.listdir(recyclable_dir)),
    len(os.listdir(non_recyclable_dir))
]

plt.figure()
plt.bar(labels, counts)
plt.title("Class Distribution (Cleaned Dataset)")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.show()

# Recyclable and non-recylable examples

import random
from PIL import Image

def show_samples(folder, title):
    images = random.sample(os.listdir(folder), 6)

    plt.figure(figsize=(8, 6))
    for i, img_name in enumerate(images):
        img = Image.open(os.path.join(folder, img_name))
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.show()

show_samples(recyclable_dir, "Recyclable")
show_samples(non_recyclable_dir, "Non-Recyclable")
