# Diffusion Sandbox

This repository contains a small experimental setup for training text-conditioned diffusion models. The code uses a convolutional VAE to encode images into spatial latents and a UNet trained with noise scheduling from the `diffusers` library. A large notebook demonstrates the full workflow on the CelebA dataset.

## Repository Structure

- **`CelebaWithAttributes.py`** – PyTorch dataset that loads CelebA images and produces a comma-separated list of positive attributes for each image.
- **`SpatialVAE.py`** – Implementation of a spatial VAE with helper function `train_vae`.
- **`train.py`** – Training utilities for the diffusion UNet, both unconditional and text-conditioned.
- **`utils.py`** – Helper for computing CLIP text embeddings used for conditioning.
- **`diffusing.ipynb`** – Jupyter notebook showing end-to-end training.
- **`requirements.txt`** – Python dependencies.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the notebook `diffusing.ipynb` to see how the dataset, VAE, and diffusion training fit together. Adjust the image and CSV paths for your environment.
3. You can also import the modules directly and use the training functions from `train.py` in your own scripts.

## Dataset

The provided dataset class expects a directory of images and a CSV file of attributes. Each sample returns the image tensor and a text string describing the attributes that are present. This text is then embedded using CLIP and fed into the diffusion model.

## Notes

This code is a minimal sandbox for learning how VAEs and diffusion models interact. It is not optimized for large-scale training but provides a clear starting point for experimentation.
