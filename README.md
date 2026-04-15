**DS 4002, Spring 2026** <br>
**Group Name**: [Your Group Name] <br>
**Group Leader:** [Leader Name] <br>
**Members:** [Member 1], [Member 2], [Member 3]

# Project 3: Binary Waste Classification Using CNNs and the RealWaste Dataset

The goal of this project is to explore whether a convolutional neural network can accurately determine whether a piece of waste is recyclable or non-recyclable based on images from the RealWaste dataset. We build a binary image classifier that predicts Recyclable vs. Non-Recyclable from real-world landfill images. We first build a baseline custom CNN, then compare it against transfer learning from a pretrained MobileNetV2. Although the original dataset has nine material classes, we convert it into a binary target to align with the practical goal of automated waste sorting. Model performance is evaluated using accuracy and F1 score.

## Software and Platform
- Utilized Google Colab's Python notebook to complete the code
- Must install torch, torchvision, Pillow, scikit-learn, matplotlib, and seaborn packages
- This code runs on either Windows or Mac systems (developed on Google Colab)

## Documentation Map
Main Branch Folders
- Data
  - RealWaste/ (original dataset, nine material class folders)
  - CleanedDataset/ (remapped to Recyclable / Non_Recyclable)
  - ResizedDataset/ (all images resized to 224x224)
  - SplitDataset/ (train / val / test folders, 80/10/10 split)
- Scripts
  - MI3_Project_3.ipynb (preprocessing, model training, and evaluation)

## Analysis Instructions

Preprocessing the Data
- Update the `data_dir`, `base_dir`, and `resized_base_dir` path variables in Cells 1–3 and 8 to match your Google Drive folder structure.
- Run Cells 3–6 to remap the nine original classes into two folders: Recyclable (Cardboard, Paper, Glass, Metal) and Non-Recyclable (Miscellaneous Trash, Textile Trash, Food Organics, Vegetation). Note: the Plastic class is excluded.
- Run Cell 8 to resize all images to 224×224 and save to ResizedDataset/.
- Run Cell 9 to randomly shuffle and split images into train (80%), val (10%), and test (10%) sets. To fix the split for reproducibility, add `random.seed(42)` at the top of Cell 9.

Modeling Steps
1. Run Cell 10 to define image transforms and DataLoaders. Add augmentation transforms (RandomHorizontalFlip, RandomVerticalFlip, RandomRotation) to the training transform only.
2. Run Cell 12 to load a pretrained MobileNetV2 and replace the final layer with a 2-class output head. (Cell 11 defines an unused baseline CNN — skip it.)
3. Run Cell 13 to train the model for up to 10 epochs using Adam (lr=0.001) and cross-entropy loss. Early stopping triggers after 3 epochs without improvement in validation loss.
4. Run Cell 14 to evaluate on the test set and print precision, recall, F1, and accuracy.
5. Run Cell 15 to plot the confusion matrix heatmap.
