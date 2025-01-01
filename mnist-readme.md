# MNIST Digit Recognition with PyTorch

A simple CNN (Convolutional Neural Network) implementation for recognizing handwritten digits using the MNIST dataset and PyTorch.

## Prerequisites

Before running this code, make sure you have the following installed:

```bash
python >= 3.6
pytorch
torchvision
matplotlib
tqdm (optional)
```

You can install the required packages using pip:

```bash
pip install torch torchvision matplotlib tqdm
```

## Project Structure

The project consists of a single Python file (`handwritten_digits_mnist.py`) that includes:
- Data loading and preprocessing
- CNN model architecture
- Training loop
- Model saving functionality
- Testing/prediction visualization

## Features

- Data split into training (80%) and validation (20%) sets
- Convolutional Neural Network with:
  - 2 convolutional layers
  - MaxPooling
  - Dropout for regularization
  - 2 fully connected layers
- Training with CrossEntropy loss and Adam optimizer
- Model performance visualization
- Ability to save both full model and state dict
- Testing function with prediction visualization

## Usage

1. Run the training:
```python
model_training()
```

2. Save the model:
```python
# To save just the state dict
save_model(full_model=False)

# To save the entire model
save_model(full_model=True)
```

3. Test the model on a specific image:
```python
# Test image at index x (x can be any number between 0 and len(mnist_testset))
testing(x, model)
```

## Model Architecture

- Input Layer: 1 channel (grayscale images)
- Conv1: 32 filters, 3x3 kernel
- Conv2: 64 filters, 3x3 kernel
- MaxPool: 2x2
- Fully Connected 1: 9216 -> 128
- Dropout: 0.5
- Fully Connected 2: 128 -> 10 (output)

## Hyperparameters

- Batch Size: 64
- Learning Rate: 0.001
- Epochs: 12
- Dropout Rate: 0.5

## Note

The model automatically downloads the MNIST dataset on first run. Make sure you have an internet connection and sufficient disk space for the dataset.
