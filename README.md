Cat ID Photo Generator
Overview

Cat ID Photo Generator is a GAN-based project that generates standard ID-style cat portraits from cat head images.

Due to the lack of publicly available paired datasets, we generated about 100 image pairs using other generators. Basic data augmentation (e.g., horizontal flipping) was applied, but the dataset remains small.

We chose pix2pix over CycleGAN since paired images are required and CycleGAN is less suitable for such limited data.

Features

Pix2pix Framework: Image-to-image translation with paired training data.

Generator: UNet-based (UNetGenerator) for detailed reconstruction.

Discriminator: PatchGAN (PatchGANDiscriminator) focusing on local image patches.

Training Stabilization: Gradient penalty improves adversarial training stability.

Modular Code: Easy to swap datasets, adjust network layers, or extend functionalities.

This project represents a logical extension of our previous DCWGAN-GP work, combining prior experience with a pix2pix framework for improved performance and stability.

Directory Structure
Cat_ID_Photo_Generator/
├─ cat_head/                 # Original cat head images
├─ cat_head_enhanced/        # Enhanced cat head images
├─ cat_head_selection/       # Selected subset of cat head images for training
├─ cat_id/                   # Original cat ID photos (training targets)
├─ cat_id_enhanced/          # Enhanced cat ID photos
├─ version1_pix2pix/         # Version 1 of pix2pix with full training code and results
│   ├─ src/                  # Training code and network definitions
│   │   ├─ generator.py
│   │   ├─ discriminator.py
│   │   └─ train.py
│   ├─ models/               # Trained models
│   └─ output/               # Generated cat ID photos
├─ requirements.txt          # Python dependencies
└─ README.md                 # Project overview and instructions

Installation
git clone <repo_url>
cd Cat_ID_Photo_Generator
pip install -r requirements.txt

Usage

Prepare data: Place your cat head images in cat_head/.

Train the model:

python v1_pix2pix/src/train.py


Generate cat ID photos:

python version1_pix2pix/src/generate.py --input cat_head/your_cat.jpg --output version1_pix2pix/output/

	
Future Improvements

Expand dataset size to explore CycleGAN for unpaired training.

Integrate super-resolution networks for higher clarity.

Support more cat breeds, backgrounds, and higher-resolution outputs.

Technical Notes

Pix2pix implementation leverages UNetGenerator + PatchGANDiscriminator.

Gradient penalty inspired by DCWGAN-GP enhances training stability.

Modular architecture allows easy experimentation with network depth, loss functions, and input/output sizes.

