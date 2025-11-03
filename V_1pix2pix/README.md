
Cat ID pthot pix2pix Project
Project Overview

This project implements a Cat Face GAN using a UNet-based generator and a PatchGAN discriminator. The goal is to generate high-quality cat ID images from cat head images, with a focus on realistic facial features. The generator learns to map cat head images (plus noise) to cat ID images, while the discriminator evaluates the realism of generated images using a PatchGAN approach.

The project also calculates the FID (Fréchet Inception Distance) to monitor the quality of generated images over training.

Directory Structure
project_root/
│
├── cat_head_enhanced/         # Enhanced cat head images (input)
├── cat_id_enhanced/           # Enhanced cat ID images (target output)
│
├── V_1pix2pix/
│   ├── generated/             # Generated images during training
│   ├── temp_fake/             # Temporary folder for saving fake images for FID calculation
│   ├── temp_real/             # Temporary folder for saving real images for FID calculation
│   ├──pix2pix.py              # Main training script (generator, discriminator, dataset, training loop)
│   ├──README.md               # This readme
└── README.md                  # Project README

Dependencies

Make sure you have the following Python packages installed:

Python 3.8+
torch
torchvision
Pillow
numpy
matplotlib
pytorch-fid


Install dependencies via pip:

pip install torch torchvision pillow numpy matplotlib pytorch-fid

Data Preparation

Collect cat head images and corresponding cat ID images.

Organize them in folders:

cat_head_enhanced/

cat_id_enhanced/

Images will be automatically resized to 128x128 during training.

Ensure that filenames in both folders are aligned, so the generator receives correct pairs.

Model Architecture
Generator (UNet)

Input: Cat head image (3 channels) + random noise (3 channels) → 6 channels

Encoder: 5 convolutional layers with increasing channels

Decoder: 5 transposed convolutional layers with skip connections

Output: 6-channel image, where the last 3 channels are the generated cat ID

Discriminator (PatchGAN)

Input: 6-channel image (cat head + cat ID)

4 convolutional layers + 1 final conv layer to output patch realism map

Uses LeakyReLU and BatchNorm (except first layer)

Loss Functions

Generator Loss: WGAN adversarial loss + L1 loss between generated and real images

Discriminator Loss: WGAN loss + gradient penalty (WGAN-GP)

Gradient Penalty: Ensures Lipschitz constraint for WGAN

Training
Hyperparameters
num_epochs = 500
batch_size = 20
lambda_L1 = 1       # L1 loss weight
lambda_GP = 10      # Gradient penalty weight
learning_rate = 0.0001 (RMSProp)

Steps

Load dataset and create DataLoader.

For each batch:

Generate fake images with the generator.

Update discriminator with real and fake images.

Update generator with adversarial + L1 loss.

Save generated images every 5 epochs for visualization.

Compute FID to evaluate image quality.

Clear temporary fake images folder after FID calculation.

How to Run
python cat_gan.py


Training images will be saved to V_1pix2pix/generated/.

Temporary real/fake images for FID will be saved to temp_real/ and temp_fake/.

Results

Example of generated image grid:

FID is printed every 5 epochs to monitor progress.

Notes

The generator input consists of the cat head image concatenated with random noise.

L1 loss encourages the generated images to resemble real cat IDs.

PatchGAN discriminator focuses on local patches to improve realism.

You can experiment with adding more layers or using a CNN-based generator for higher-quality images.

