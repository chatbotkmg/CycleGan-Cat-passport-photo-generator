
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shutil
import time
import torch.nn.init as init
from pytorch_fid import fid_score  


# ===== UNet Generator =====
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(6, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 1024)

        # Decoder
        self.decoder1 = self.deconv_block(1024, 512, dropout=True)
        self.decoder2 = self.deconv_block(1024, 256, dropout=True)
        self.decoder3 = self.deconv_block(512, 128)
        self.decoder4 = self.deconv_block(256, 64)
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(128, 6, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    # Convolution block
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    # Transposed convolution (deconv) block
    def deconv_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    # Forward pass
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        # Decoder with skip connections
        dec1 = self.decoder1(enc5)
        dec1 = torch.cat([dec1, enc4], dim=1)

        dec2 = self.decoder2(dec1)
        dec2 = torch.cat([dec2, enc3], dim=1)

        dec3 = self.decoder3(dec2)
        dec3 = torch.cat([dec3, enc2], dim=1)

        dec4 = self.decoder4(dec3)
        dec4 = torch.cat([dec4, enc1], dim=1)

        out = self.decoder5(dec4)
        return out

    # Initialize weights
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)


# Instantiate generator and initialize weights
model = UNetGenerator()
model.init_weights()





# ===== PatchGAN Discriminator =====
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()

        # Convolution block
        def conv_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # PatchGAN model
        self.model = nn.Sequential(
            *conv_block(6, 64, normalize=False),
            *conv_block(64, 128),
            *conv_block(128, 256),
            *conv_block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1, bias=False)
        )

    # Forward pass
    def forward(self, x):
        return self.model(x)

    # Initialize weights
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')


# Instantiate discriminator and initialize weights
model = PatchGANDiscriminator()
model.init_weights()



# ===== Cat Dataset =====
class CatDataset(Dataset):
    def __init__(self, cat_head_dir, cat_id_dir, img_size=(128, 128), noise_dim=3):
        self.cat_head_dir = cat_head_dir
        self.cat_id_dir = cat_id_dir
        self.img_size = img_size
        self.noise_dim = noise_dim
        self.cat_head_files = sorted(os.listdir(cat_head_dir))
        self.cat_id_files = sorted(os.listdir(cat_id_dir))

    def __len__(self):
        return len(self.cat_head_files)

    # Get item for DataLoader
    def __getitem__(self, idx):
        cat_head_path = os.path.join(self.cat_head_dir, self.cat_head_files[idx])
        cat_id_path = os.path.join(self.cat_id_dir, self.cat_id_files[idx])

        cat_head = Image.open(cat_head_path).convert('RGB')
        cat_id = Image.open(cat_id_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        cat_head = transform(cat_head)
        cat_id = transform(cat_id)

        # Concatenate cat head and ID image to 6-channel image
        combined_image = torch.cat((cat_head, cat_id), dim=0)

        # Generate random noise as input for generator
        noise = torch.randn(self.noise_dim, cat_head.size(1), cat_head.size(2))
        generator_input = torch.cat((cat_head, noise), dim=0)

        return generator_input, combined_image, cat_id


# Generator loss function
def generator_loss(D, G, real_images, fake_images, lambda_L1):
    # Adversarial loss (WGAN generator loss)
    adversarial_loss = -torch.mean(D(fake_images))

    # L1 loss between generated image and real image
    L1_loss = F.l1_loss(fake_images, real_images)

    # Total generator loss
    total_loss = adversarial_loss + lambda_L1 * L1_loss
    return total_loss


# Discriminator loss function
def discriminator_loss(D, real_images, fake_images, lambda_GP):
    real_loss = torch.mean(D(real_images))
    fake_loss = torch.mean(D(fake_images))
    gradient_penalty = calculate_gradient_penalty(D, real_images, fake_images)
    total_loss = fake_loss - real_loss + lambda_GP * gradient_penalty
    return total_loss


# Gradient penalty for WGAN-GP
def calculate_gradient_penalty(D, real_images, fake_images):
    alpha = torch.rand((real_images.size(0), 1, 1, 1)).to(real_images.device)
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images
    interpolated_images.requires_grad_(True)
    pred = D(interpolated_images)
    gradients = torch.autograd.grad(outputs=pred, inputs=interpolated_images,
                                    grad_outputs=torch.ones_like(pred),
                                    create_graph=True, retain_graph=True)[0]
    gradient_penalty = torch.mean((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2)
    return gradient_penalty


# Directories for saving images
generated_dir = "/kaggle/working/V_1pix2pix/generated/"
fake_image_dir = "/kaggle/working/V_1pix2pix/temp_fake/"
real_image_dir = "/kaggle/working/V_1pix2pix/temp_real/"
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(fake_image_dir, exist_ok=True)
os.makedirs(real_image_dir, exist_ok=True)


# Save image grid
def save_image_grid(images, epoch, prefix, grid_size=3, show_first=False):
    images_np = images.permute(0,2,3,1).detach().cpu().numpy()

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for i, ax in enumerate(axes.flatten()):
        if i < images_np.shape[0]:
            ax.imshow((images_np[i]+1)/2)
            ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join(generated_dir, f'{prefix}_epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()

    if show_first:
        plt.imshow((images_np[0]+1)/2)
        plt.axis('off')
        plt.show()

    print(f"Saved and displayed generated images to {save_path}")


# Clear fake images folder
def clear_fake_images():
    if os.path.exists(fake_image_dir):
        shutil.rmtree(fake_image_dir)
    os.makedirs(fake_image_dir, exist_ok=True)


# Optimizers and device
beta1 = 0.5
beta2 = 0.999
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = UNetGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.0001)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.0001)



# Training loop
num_epochs = 500
lambda_L1 = 1
lambda_GP = 10
batch_size = 20

dataset = CatDataset("/kaggle/working/cat_head_enhanced/", "/kaggle/working/cat_id_enhanced/")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


d_loss_total = 0
g_loss_total = 0
correct_generated = 0
total_images = 0

# Save 9 real images before training
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    for batch_idx, (real_generator_input, real_combined_image, real_cat_id) in enumerate(dataloader):

        real_generator_input = real_generator_input.to(device)
        real_combined_image = real_combined_image.to(device)

        if epoch == 0 and batch_idx == 0:
            save_image_grid(real_cat_id[:9], epoch=0, prefix='real_before_training', grid_size=3, show_first=True)
            num_images_to_save = min(9, real_cat_id.size(0))
            for i in range(num_images_to_save):
                img = transforms.ToPILImage()(real_cat_id[i])
                img.save(os.path.join(real_image_dir, f"real_cat_id_{i+1}.jpg"))

        fake_images = generator(real_generator_input)

        # Train discriminator
        optimizer_D.zero_grad()
        real_loss = discriminator_loss(discriminator, real_combined_image, fake_images.detach(), lambda_GP)
        real_loss.backward()
        optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad()
        gen_loss = generator_loss(discriminator, generator, real_combined_image, fake_images, lambda_L1)
        gen_loss.backward()
        optimizer_G.step()

        # Accumulate losses and accuracy
        d_loss_total += real_loss.item()
        g_loss_total += gen_loss.item()
        correct_generated += (fake_images > 0).sum().item()
        total_images += real_combined_image.size(0)

    epoch_time = time.time() - epoch_start_time
    epoch_d_loss = d_loss_total / len(dataloader)
    epoch_g_loss = g_loss_total / len(dataloader)
    epoch_accuracy = 100 * correct_generated / total_images

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}, "
          f"Generator Accuracy: {epoch_accuracy:.2f}%, "
          f"Time: {epoch_time:.2f}s")


    if (epoch + 1) % 5 == 0:
        fake_cat_id = fake_images[:, 3:, :, :]
        num_images_to_save = min(9, fake_cat_id.size(0))
        for j in range(num_images_to_save):
            img = transforms.ToPILImage()(fake_cat_id[j])
            img.save(os.path.join(fake_image_dir, f'fake_{epoch+1}_{j+1}.jpg'))
        
        save_image_grid(fake_cat_id.cpu(), epoch + 1, prefix='generated', grid_size=3, show_first=True)
        print(f"Saved 9 images to {fake_image_dir} for epoch {epoch+1}")

        # Calculate FID
        fid_value = fid_score.calculate_fid_given_paths([real_image_dir, fake_image_dir], batch_size=9, dims=2048, device=device)
        print(f"Epoch [{epoch+1}/{num_epochs}], FID: {fid_value:.4f}")
        
        clear_fake_images()



