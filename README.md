# CIFAR-10 Image Classification with Custom ML Architecture

This project implements a custom neural network architecture for classifying images from the CIFAR-10 dataset using intermediate blocks.

## Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal of this project is to develop and train a custom neural network architecture that can effectively classify these images.

## Dataset

CIFAR-10 contains the following classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Architecture

The model uses a custom architecture with parallel intermediate blocks, which allows for:
- Parallel feature extraction through multiple convolutional paths
- Adaptive feature fusion using learned attention weights
- Hierarchical processing with batch normalization
- Improved gradient flow through residual connections
- Enhanced model capacity while maintaining computational efficiency

### Key Components

1. **Intermediate Block**:
   - Multiple parallel convolutional paths
   - Each path contains:
     - Two convolutional layers with batch normalization and ReLU activation
     - Adaptive average pooling
   - Attention mechanism to weight different parallel paths
   - Softmax-based feature fusion

2. **Main Network**:
   - Stack of intermediate blocks
   - Global average pooling
   - Dropout (p=0.7) for regularization
   - Final classification layer

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

## Installation

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## Usage

Clone the repository:
```bash
git clone https://github.com/shahinmv/cifar10-custom-architecture.git
```

## Model Structure

The custom architecture consists of:
- Input layer (32x32x3 images)
- Multiple intermediate blocks, each containing:
  - Parallel convolutional paths
  - Batch normalization layers
  - ReLU activation functions
  - Attention-based feature fusion
- Global average pooling
- Dropout layer (p=0.7)
- Output layer (10 classes)

## Training

The model is trained using:
- Cross-entropy loss
- Adam optimizer
- Learning rate scheduling
- Data augmentation techniques

## Evaluation

The model's performance is evaluated using:
- Training accuracy
- Validation accuracy
- Test accuracy
- Confusion matrix
- Classification report

## Results

The model's performance metrics and visualizations are saved in the results directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 