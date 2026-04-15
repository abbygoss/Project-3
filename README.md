**DS 4002, Spring 2026** <br>
**Group Name**: Freenor's Fourth Years <br>
**Group Leader:** Abby Goss <br>
**Members:** Christina Barton, Abby Goss, Rohan Kohli

# Project 3: Binary Waste Classification Using CNNs and the RealWaste Dataset

The goal of this project is to explore whether a convolutional neural network can accurately determine whether a piece of waste is recyclable or non-recyclable based on images from the RealWaste dataset. We build a binary image classifier that predicts Recyclable vs. Non-Recyclable from real-world landfill images. We first build a baseline custom CNN, then compare it against transfer learning from a pretrained MobileNetV2. Although the original dataset has nine material classes, we converted it into a binary target to align with the practical goal of automated waste sorting. Model performance is evaluated using accuracy and F1 score.

## Software and Platform
- Utilized Google Colab's Python notebook to complete the code
- Must install torch, torchvision, Pillow, scikit-learn, matplotlib, and seaborn packages
- This code runs on either Windows or Mac systems (developed on Google Colab)

## Documentation Map
Main Branch Folders
- Data
  - Link to Datasets (RealWaste/CleanedDataset/ResizedDataset/SplitDataset)
  - Data Appendix 
- Scripts
  - EDA Project 3.ipynb (exploratory data analysis)
  - MI3_Project_3.ipynb (preprocessing, model training, and evaluation)
- Output

## Analysis Instructions

Preparing the Data

- Mount your Google Drive in Colab and update the data_dir, base_dir, and resized_base_dir path variables to point to wherever you stored the RealWaste folder on your Drive.
- The original dataset has nine material class folders. Remap these into two binary categories: Recyclable (Cardboard, Paper, Glass, Metal) and Non-Recyclable (Miscellaneous Trash, Textile Trash, Food Organics, Vegetation) by copying images into new Recyclable/ and Non_Recyclable/ subfolders. The Plastic class is excluded entirely.
- Resize all images to 224×224 pixels using PIL and save them to a new ResizedDataset/ folder. This standardizes input dimensions for the CNN.
- Shuffle the resized images randomly and split them into training (80%), validation (10%), and test (10%) sets, saving each split into its own subfolder under SplitDataset/. To ensure a reproducible split, add random.seed(42) before the shuffle call.

Modeling Steps

1. Define separate image transforms for training and evaluation. For the training set, apply random horizontal flips, vertical flips, and small rotations as data augmentation to help the model generalize. For validation and test sets, apply only normalization using ImageNet mean and standard deviation values. Wrap each split in a PyTorch DataLoader with a batch size of 32, and set shuffle=True for the training loader only.
2. Define the baseline CNN architecture: two convolutional layers (16 and 32 filters) each followed by ReLU activation and max pooling, then a fully connected layer of 128 units, and a final output layer with 2 classes. This is a lightweight model trained from scratch to serve as a performance baseline.
3. Train the baseline CNN for up to 10 epochs using the Adam optimizer (learning rate 0.001) and cross-entropy loss. After each epoch, compute validation loss. If validation loss does not improve for 3 consecutive epochs, stop training early to prevent overfitting.
4. Evaluate the trained model on the held-out test set. Generate predictions by taking the argmax of the model's output logits, then compute accuracy, precision, recall, and F1 score using a classification report.
5. Plot a confusion matrix heatmap showing true vs. predicted labels for Recyclable and Non-Recyclable classes to visualize where the model makes errors.
6. Repeat steps 2–5 using a pretrained MobileNetV2 instead of the baseline CNN. Load the pretrained weights, replace the final classifier layer with a 2-class linear layer, and train using the same optimizer settings and early stopping criteria. Compare the accuracy and F1 scores of the baseline CNN vs. MobileNetV2 to assess whether transfer learning improves classification performance on the RealWaste dataset.
