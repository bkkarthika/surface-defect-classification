# Surface Defect Classification using Transfer Learning

## Project Overview

This project focuses on classifying steel surface defects using **deep learning and transfer learning**. A pretrained **MobileNetV2 model with ImageNet weights** is adapted to classify six different types of steel surface defects.

The project demonstrates an end-to-end image classification workflow, including dataset exploration, image preprocessing, transfer learning, model training, and performance evaluation.

---

## Business Problem

In steel manufacturing, identifying surface defects accurately and consistently is an important part of quality control.

Manual inspection can be time-consuming and may be affected by human variability. An image classification model can assist quality-control teams by automatically identifying different types of surface defects from inspection images.

The objective of this project is to develop a model that can classify steel surface images into their respective defect categories.

---

## Objective

The main objectives of this project are to:

- Understand and explore the surface defect dataset.
- Prepare grayscale images for an ImageNet-pretrained model.
- Apply appropriate image preprocessing and transformations.
- Use transfer learning with MobileNetV2.
- Freeze the pretrained feature extractor and train a task-specific classifier.
- Evaluate the model using accuracy, precision, recall and F1-score.
- Analyse classification errors using a confusion matrix.
- Assess the practical business value of the solution.

---

## Dataset

The dataset contains images belonging to six steel surface defect categories:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

The training dataset contains **240 images per class**, giving a balanced training dataset of **1,440 images**.

The validation set contains **60 images per class**, giving a total of **360 validation images**.

---

## Approach

The project follows the following workflow:

```text
Dataset
   ↓
Dataset Exploration
   ↓
Image Preprocessing
   ↓
Train / Validation Data Loaders
   ↓
MobileNetV2 with ImageNet Weights
   ↓
Freeze Feature Extractor
   ↓
Replace Classification Head
   ↓
Train Model
   ↓
Evaluate on Validation Set
   ↓
Classification Report
   ↓
Confusion Matrix
   ↓
Business Insights
```

---

## Image Preprocessing

The original images are grayscale, while the pretrained MobileNetV2 model expects three-channel input.

The following preprocessing steps are applied:

- Convert grayscale images to 3 channels.
- Resize images to **224 × 224 pixels**.
- Convert images to PyTorch tensors.
- Normalize images using the **ImageNet mean and standard deviation**.

The resulting image tensor has the shape:

```text
[3, 224, 224]
```

These transformations make the images compatible with the pretrained MobileNetV2 architecture.

---

## Transfer Learning

### Model Selection

**MobileNetV2** with pretrained **ImageNet weights** was selected for this classification task.

MobileNetV2 provides a pretrained convolutional feature extractor while keeping the model relatively lightweight compared with larger image classification architectures.

### Freezing Strategy

The pretrained feature extraction layers are frozen:

```python
for param in model.features.parameters():
    param.requires_grad = False
```

This preserves the visual features learned from ImageNet and reduces the number of parameters that need to be trained.

The original ImageNet classifier is replaced with a task-specific classifier for the six surface defect classes.

The new classification head consists of:

```text
Linear
   ↓
ReLU
   ↓
Dropout
   ↓
Linear
```

Only the new classifier is trained.

### Parameter Summary

| Parameter | Value |
|---|---:|
| Total parameters | 2,553,350 |
| Trainable parameters | 329,478 |
| Frozen parameters | 2,223,872 |
| Trainable percentage | 12.90% |

---

## Model Training

The model was trained using:

| Configuration | Value |
|---|---|
| Model | MobileNetV2 |
| Pretrained weights | ImageNet |
| Feature extractor | Frozen |
| Epochs | 10 |
| Batch size | 32 |
| Number of classes | 6 |

The training and validation loss and accuracy were tracked during training to analyse model learning and generalization.

---

## Evaluation

The trained model was evaluated on **360 validation images**.

The evaluation includes:

- Validation accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Misclassification analysis

### Final Results

| Metric | Result |
|---|---:|
| Training accuracy | **100.00%** |
| Validation accuracy | **98.33%** |
| Validation images | **360** |
| Misclassified images | **6** |
| Train–validation accuracy gap | **1.67 percentage points** |
| Macro F1-score | **0.983** |

The model correctly classified **354 out of 360 validation images**.

### Class-wise Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Crazing | 1.000 | 1.000 | 1.000 |
| Inclusion | 1.000 | 0.917 | 0.957 |
| Patches | 1.000 | 1.000 | 1.000 |
| Pitted Surface | 0.968 | 1.000 | 0.984 |
| Rolled-in Scale | 0.984 | 1.000 | 0.992 |
| Scratches | 0.952 | 0.983 | 0.967 |

The confusion matrix is used to identify which defect categories are most frequently confused with one another.

---

## Key Insights

- The model achieved **98.33% validation accuracy**, indicating strong classification performance.
- Only **6 of 360 validation images** were misclassified.
- The training and validation accuracy gap was only **1.67 percentage points**, indicating good generalization.
- The training dataset was balanced, with **240 images per defect class**.
- Crazing and patches achieved perfect precision, recall and F1-score on the validation set.
- Inclusion and scratches showed relatively more classification errors and could be areas for further improvement.
- Transfer learning allowed the model to achieve strong performance while training only **12.90% of the total model parameters**.

---

## Business Value

An automated surface defect classification system can support industrial quality-control processes by assisting inspectors in identifying defective products.

Potential business benefits include:

- Faster inspection of steel surfaces.
- More consistent defect identification.
- Reduced dependence on manual inspection alone.
- Early identification of defective products.
- Potential integration into automated visual inspection systems.
- Lower computational requirements compared with training a large deep learning model from scratch.

The model can therefore serve as a **decision-support tool for quality inspection**, with further testing required on real-world production-line images before deployment.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PIL
- MobileNetV2
- Transfer Learning
- Image Classification

---

## Project Structure

```text
surface-defect-classification/
│
├── README.md
├── Transfer_Learning_Surface_Defect_<karthika>.ipynb
└── .gitignore
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd surface-defect-classification
```

### 2. Install the required Python packages

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow
```

### 3. Prepare the dataset

Download the surface defect dataset and place it in the directory structure expected by the notebook.

The notebook contains the dataset loading and preprocessing steps.

### 4. Open the notebook

Open:

```text
Transfer_Learning_Surface_Defect_<karthika>(1).ipynb
```

using Jupyter Notebook, JupyterLab, or VS Code.

### 5. Run the notebook

Run the cells sequentially to reproduce:

- Dataset exploration
- Image preprocessing
- Data loaders
- Transfer learning model
- Model training
- Validation
- Classification report
- Confusion matrix
- Prediction analysis

---

## Future Improvements

Potential future improvements include:

- Evaluate the model on a larger and more diverse dataset.
- Test additional pretrained architectures.
- Investigate the misclassified images in greater detail.
- Experiment with additional training strategies.
- Validate the model using real production-line inspection images.
- Explore deployment on resource-constrained inspection devices.

---

## Project Outcome

The project demonstrates that a pretrained MobileNetV2 model can be effectively adapted for steel surface defect classification using transfer learning.

The final model achieved **98.33% validation accuracy**, with only **6 misclassifications out of 360 validation images**, while only **12.90% of the model parameters were trainable**.

This project demonstrates practical experience with **computer vision, transfer learning, PyTorch, model evaluation, and applying machine learning to an industrial quality-control problem**.