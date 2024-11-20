# Conditional Diffusion Model with Classifier Guidance

This project implements a conditional diffusion model with classifier guidance for generating butterfly images. The implementation includes a U-Net architecture for both the diffusion model and classifier, incorporating attention mechanisms and residual blocks.

## Features

- Conditional image generation using diffusion models
- Classifier guidance for controlled generation
- Multi-head attention mechanisms
- Residual blocks with time embeddings
- Sinusoidal positional encoding
- Custom dataset handling for butterfly images

## Model Architecture

### Key Components

- **UNet**: Base diffusion model with:
  - Encoder-decoder architecture
  - Multi-head attention layers
  - Residual blocks with time embeddings
  - Sinusoidal positional encoding

- **UNet_classifier**: Classification model with:
  - Similar architecture to base UNet
  - Modified output head for classification
  - 75 class output for butterfly species

### Training

The model uses:
- Adam optimizer
- MSE loss for diffusion
- Cross-entropy loss for classification
- Linear noise scheduler
- Combined training of both diffusion and classifier models

## Usage

### Dependencies

```python
torch
torchvision
numpy
pandas
PIL
tqdm
matplotlib
torchmetrics
