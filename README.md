# Neural-Network-from-Scratch

# Neural Network from Scratch for Medical Image Segmentation

## Overview
This repository presents a fully customized neural network built from scratch using only NumPy for medical image segmentation. This task is vital in biomedical engineering for applications such as tumor detection, organ delineation, and disease diagnosis. 

This project demonstrates a strong grasp of deep learning fundamentals—such as forward/backward propagation, convolution operations, and optimization—without relying on high-level libraries like TensorFlow or PyTorch. Focusing on MRI brain tumor segmentation, it reflects my research interest in AI-driven tools for computer-aided diagnostics, which I aim to explore further in a PhD program in biomedical engineering.

## Project Objectives
- Build a convolutional neural network (CNN) from scratch for brain tumor segmentation.
- Illustrate deep learning principles including convolutional layers, ReLU, binary cross-entropy, and backpropagation.
- Apply neural networks to real-world biomedical problems.
- Offer clear visualizations and documentation to explain model behavior and performance.

## Dataset
We utilize the **Brain MRI Segmentation Dataset** from Kaggle:
- **Images:** T2-weighted brain MRI scans in PNG format (256x256).
- **Masks:** Binary segmentation masks highlighting low-grade glioma (LGG) tumors.
- **Data Size:** Scans from ~110 patients, multiple slices per patient.

> **Note:** Dataset setup instructions are in the **Setup** section below.

## Methodology
### Neural Network Architecture
The model follows a simplified U-Net-like structure:
- **Input Layer:** Accepts grayscale MRI (256x256x1).
- **Conv Layers:** 3x3 kernels with ReLU activation.
- **Pooling:** 2x2 max pooling layers.
- **Upsampling:** Nearest-neighbor upsampling.
- **Output Layer:** Sigmoid-activated (256x256x1) binary mask.
- **Loss:** Binary cross-entropy.
- **Optimizer:** Mini-batch gradient descent with momentum.

### Preprocessing
- **Normalization:** Pixel scaling to [0, 1].
- **Augmentation:** Random flips/rotations.
- **Split:** 80% training / 20% test.

### Training
- **Epochs:** 50
- **Batch Size:** 8
- **Learning Rate:** 0.001
- **Validation:** On held-out test set

### Evaluation Metrics
- **Dice Coefficient**
- **Intersection over Union (IoU)**
- **Pixel-wise Accuracy**

## Repository Structure
```
neural-network-from-scratch/
├── data/                     # Dataset storage
├── src/                      # Core scripts
│   ├── preprocessing.py      # Preprocessing functions
│   ├── model.py              # CNN architecture
│   ├── train.py              # Model training
│   ├── evaluate.py           # Metrics and evaluation
│   └── utils.py              # Visualization helpers
├── notebooks/                # Notebooks
│   └── demo.ipynb            # End-to-end demo
├── results/                  # Output images & graphs
│   ├── loss_curve.png
│   ├── dice_curve.png
│   └── sample_predictions/   # Sample outputs
├── docs/
│   └── project_report.md     # Full documentation
├── README.md
├── requirements.txt
└── LICENSE
```


