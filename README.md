# Handwritten Mathematical Symbol Recognition

An end-to-end machine learning project that trains a convolutional neural network (CNN) to recognize handwritten mathematical symbols and deploys it in an interactive drawing-based application for real-time classification.

---

## Project Overview

Handwritten mathematical symbols vary widely in shape, size, and writing style, making reliable recognition a challenging computer vision problem.  
This project addresses that challenge by:

- Training a convolutional neural network to classify handwritten digits, variables, and operators
- Building a real-time desktop application that allows users to draw symbols and receive predictions instantly
- Exploring early-stage multi-symbol expression recognition

The result is a complete pipeline from data preprocessing and model training to deployment in an interactive user-facing tool.

---

## Problem Statement

Unlike typed math expressions, handwritten symbols are highly inconsistent due to:
- Individual writing styles
- Stroke thickness and pressure
- Rotation and placement variability
- Incomplete or ambiguous strokes

Traditional rule-based approaches struggle with this variability. This project demonstrates how a CNN can learn spatial features directly from raw pixel data to achieve high classification accuracy.

---

## Dataset

This project uses the **Handwritten Math Symbols** dataset from Kaggle:

https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols

**Important:**  
The dataset is **not included** in this repository due to size and licensing considerations.

### Dataset Structure

After downloading, the dataset should be placed in the project root with the following structure:
handwritten math symbols/
├── add/
├── dec/
├── div/
├── eq/
├── mul/
├── sub/
├── x/
├── y/
├── z/
└── 0–9/

---

## Model Architecture

The classifier is a custom convolutional neural network built with PyTorch:

- 4 convolutional blocks  
  - Convolution → Batch Normalization → ReLU → Max Pooling
- Fully connected classifier
  - 512-unit dense layer with dropout
  - 19-class output layer

This architecture balances performance and simplicity while maintaining strong generalization.

---

## Training Details

- **Framework:** PyTorch
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy Loss
- **Epochs:** 35
- **Batch Size:** 64
- **Regularization:** Dropout + weight decay
- **Train/Test Split:** 80% / 20%

Training metrics (loss and accuracy) are logged and visualized across epochs.

---

## Results

- **Test Accuracy:** **99.45%**
- **Performance Highlights:**
  - Strong diagonal structure in the confusion matrix
  - Only 11 misclassifications across 19 classes
  - Errors primarily caused by ambiguous handwriting or clipped strokes

Misclassified examples were analyzed to identify limitations related to symbol similarity, stroke faintness, and preprocessing artifacts.

---

## Interactive Symbol Classifier

A desktop GUI application was built using Tkinter that allows users to:

1. Draw a symbol on a canvas  
2. Run the trained CNN in real time  
3. Receive a predicted label with confidence score  

The application uses the same preprocessing pipeline as training, ensuring consistent inference behavior.

---

## Math Expression Solver (Prototype)

As an exploratory extension, a prototype math expression solver was implemented:

- Segments multiple handwritten symbols from a single canvas
- Classifies each symbol individually
- Evaluates simple arithmetic expressions

This component demonstrates how the symbol classifier could be extended toward full handwritten math recognition, though it remains sensitive to spacing and symbol separation.

---

## Key Skills Demonstrated

- Convolutional Neural Networks (CNNs)
- Data preprocessing and augmentation
- Model evaluation and error analysis
- PyTorch model development and deployment
- Real-time inference pipelines
- GUI-based ML applications
- Reproducible ML workflows

---

## Future Improvements

- More robust multi-symbol segmentation
- Support for multi-line expressions
- Integration with symbolic math libraries
- Expanded dataset with additional symbols
- Improved handling of ambiguous or low-contrast handwriting

---

## Setup & Reproducibility

To reproduce the results:

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Download the dataset from Kaggle and place it in the project root as described above
4. Train the model:
   ```python
   final_project.py
5. Launch the interactive app:
	 ```python
   symbol_predictor.py

## Repository Structure
├── training/
│   ├── final_project.py
│   └── training_logs.csv
├── app/
│   ├── symbol_predictor.py
│   └── math_solver.py
├── assets/
│   └── figures and screenshots
├── .gitignore
└── README.md
