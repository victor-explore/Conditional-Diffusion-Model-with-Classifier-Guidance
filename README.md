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
```

### Training

```python
# Initialize models
model_0 = UNet().to(device)    # diffusion model
model_1 = UNet_classifier(num_classes=75).to(device)     # classifier model

# Train models
train_classifier_guidance(model_0, model_1, num_classes=75)
```

### Generating Images

```python
# Generate images for specific classes
generate_and_save_images(NUM_CLASSES=10, display=True)
```

## Image Generation Process

1. Start with random noise
2. Iteratively denoise using the diffusion model
3. Apply classifier guidance to steer generation
4. Generate multiple samples per class

## Model Parameters

- Number of timesteps: 1000
- Beta schedule: Linear (1e-4 to 0.02)
- Base number of filters: 64
- Embedding dimension: 32
- Image size: 128x128
- Batch size: 32

## Output

Generated images are saved in the 'conditional-sampling' directory, with multiple samples per class displayed in a grid format.

## Notes

- The model uses a custom noise scheduler for the diffusion process
- Classifier guidance scale can be adjusted for stronger/weaker conditioning
- Training includes simultaneous optimization of both diffusion and classifier models
