# RealWaste Binary Classifier

## Project Goal

The goal of this project is to explore whether a convolutional neural network (CNN) can accurately determine whether a piece of waste is recyclable or non-recyclable based on images from the RealWaste dataset. We build a binary image classifier that predicts one of two labels — **Recyclable** or **Non-Recyclable** — from real-world landfill images captured at Whyte's Gully Waste and Resource Recovery Centre in Wollongong, New South Wales, Australia.

This approach fits the RealWaste dataset because it contains real-world images with significant visual variation. CNNs are a strong choice for this task because they learn visual features such as edges, textures, and shapes directly from images, which helps distinguish recyclable materials from non-recyclable ones. We first build a baseline custom CNN, then compare it against transfer learning from a pretrained MobileNetV2. Transfer learning is well-suited here because pretrained models often generalize effectively on moderate-sized image datasets. Although the original dataset has nine material classes, we convert it into a binary target to improve model performance and align with our practical goal of automated waste sorting. Model performance is evaluated using accuracy and F1 score.

---

## Repository Contents

This repository contains the following files and folders:

```
project-root/
│
├── README.md                  # This file
│
├── MI3_Project_3.ipynb        # Main notebook: preprocessing, training, and evaluation
│
├── data/                      # (Not included — see setup instructions below)
│   ├── RealWaste/             # Original dataset, organized by material class
│   ├── CleanedDataset/        # Remapped to Recyclable / Non_Recyclable
│   ├── ResizedDataset/        # Images resized to 224x224
│   └── SplitDataset/          # Train / val / test split (80/10/10)
│       ├── train/
│       ├── val/
│       └── test/
```

---

## Section 1: Software and Platform

**Primary Software**
- Python 3.x (via Google Colab)
- Jupyter Notebook / Google Colab

**Key Packages**
The following packages are required. All are pre-installed in Google Colab; if running locally, install via `pip install <package>`:

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Model definition, training, data loading |
| `Pillow` | Image resizing |
| `scikit-learn` | Classification report and confusion matrix |
| `matplotlib` | Plotting |
| `seaborn` | Confusion matrix heatmap |

**Platform**
- Developed and tested on **Google Colab** (Linux-based cloud environment)
- Compatible with Windows, Mac, or Linux if running locally, provided Google Drive paths are updated

---

## Section 2: Project Folder Map

```
project-root/
│
├── README.md
│
├── MI3_Project_3.ipynb
│    ├── Cell 1–2:   Mount Google Drive and set data directory
│    ├── Cell 3–6:   Remap original classes → Recyclable / Non_Recyclable
│    ├── Cell 7:     Verify image counts
│    ├── Cell 8:     Resize all images to 224x224
│    ├── Cell 9:     Split into train / val / test (80/10/10)
│    ├── Cell 10:    Define transforms and DataLoaders
│    ├── Cell 11:    (Unused) Custom CNN definition
│    ├── Cell 12:    Load pretrained MobileNetV2, replace classifier head
│    ├── Cell 13:    Train with early stopping (patience=3, max 10 epochs)
│    ├── Cell 14:    Evaluate on test set, print classification report
│    └── Cell 15:    Plot confusion matrix
│
└── data/                      # Generated during preprocessing — not committed to repo
    ├── RealWaste/             # Source dataset
    ├── CleanedDataset/
    ├── ResizedDataset/
    └── SplitDataset/
```

---

## Section 3: Reproducing the Results

Follow these steps in order to reproduce the results from scratch.

### Step 1: Obtain the Dataset

Download the RealWaste dataset. It should contain the following nine class folders:

- Cardboard (461 images)
- Food Organics (411 images)
- Glass (420 images)
- Metal (790 images)
- Miscellaneous Trash (495 images)
- Paper (500 images)
- Plastic (921 images)
- Textile Trash (318 images)
- Vegetation (436 images)

Upload the dataset to your Google Drive. Note the full path — you will need it in the next step.

### Step 2: Configure the Notebook

Open `MI3_Project_3.ipynb` in Google Colab. In **Cell 1** (and again in **Cell 2**), update the `data_dir` variable to match the path where you placed the RealWaste folder on your Drive:

```python
data_dir = "/content/drive/MyDrive/YOUR_FOLDER/RealWaste"
```

Also update `base_dir` in **Cell 3** and `resized_base_dir` in **Cell 8** to point to the same parent folder.

### Step 3: Run Preprocessing (Cells 3–9)

Run cells 3 through 9 in order. These cells will:

1. Create a `CleanedDataset` folder with two subfolders — `Recyclable` and `Non_Recyclable` — and copy images into the appropriate folder based on material type. Recyclable classes are Cardboard, Paper, Glass, and Metal. Non-Recyclable classes are Miscellaneous Trash, Textile Trash, Food Organics, and Vegetation. Note: the Plastic class is excluded entirely.
2. Resize all images to 224×224 pixels and save them to a `ResizedDataset` folder.
3. Randomly shuffle and split the resized images into train (80%), validation (10%), and test (10%) sets, saved to a `SplitDataset` folder.

> **Note:** The train/val/test split uses `random.shuffle` without a fixed seed, so the exact split will differ across runs. To fix the split for reproducibility, add `random.seed(42)` at the top of Cell 9 before the `split_data` calls.

### Step 4: Define Transforms and Load Data (Cell 10)

Run Cell 10. This defines the image normalization transforms and wraps the split folders in PyTorch `DataLoader` objects. If you added data augmentation (recommended), ensure your training transform includes `RandomHorizontalFlip`, `RandomVerticalFlip`, and `RandomRotation` as described in the project notes.

### Step 5: Load the Model (Cell 12)

Run Cell 12 (skip Cell 11 — it defines an unused custom CNN). This loads a pretrained MobileNetV2 and replaces its final classification layer with a 2-class output head for binary classification.

### Step 6: Train the Model (Cell 13)

Run Cell 13. Training runs for up to 10 epochs using the Adam optimizer (learning rate 0.001) and cross-entropy loss. Early stopping is applied with a patience of 3 — if validation loss does not improve for 3 consecutive epochs, training halts automatically. Validation loss is printed after each epoch.

### Step 7: Evaluate on the Test Set (Cell 14)

Run Cell 14. The model is evaluated on the held-out test set. A full classification report is printed, including precision, recall, and F1 score for each class.

### Step 8: Plot the Confusion Matrix (Cell 15)

Run Cell 15 to generate a heatmap of the confusion matrix, showing true vs. predicted labels for Recyclable and Non-Recyclable classes.
