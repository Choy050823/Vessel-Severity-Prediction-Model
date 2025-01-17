# Step 1: Install Required Libraries
# !pip install torch torchvision pytorch-lightning pandas scikit-learn

# Step 2: Import Libraries
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 3: Load and Preprocess Your .csv File
# Upload your .csv file to Colab or mount Google Drive
from google.colab import files
uploaded = files.upload()

# Load the .csv file
file_name = list(uploaded.keys())[0]  # Replace with your file name if needed
data = pd.read_csv(file_name)

# Preprocess the data (normalize numerical columns)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.values)

# Convert to PyTorch tensors
data_tensor = torch.tensor(scaled_data, dtype=torch.float32)

# Create a DataLoader
batch_size = 64
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 4: Define the GAN
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, data_dim),
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        return self.model(data)

class GAN(pl.LightningModule):
    def __init__(self, latent_dim, data_dim, lr):
        super().__init__()
        self.generator = Generator(latent_dim, data_dim)
        self.discriminator = Discriminator(data_dim)
        self.lr = lr

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return nn.BCELoss()(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_data = batch[0]

        # Train generator
        if optimizer_idx == 0:
            z = torch.randn(real_data.shape[0], latent_dim).to(self.device)
            fake_data = self(z)
            validity = self.discriminator(fake_data)
            g_loss = self.adversarial_loss(validity, torch.ones_like(validity))
            self.log("g_loss", g_loss)
            return g_loss

        # Train discriminator
        if optimizer_idx == 1:
            validity_real = self.discriminator(real_data)
            d_loss_real = self.adversarial_loss(validity_real, torch.ones_like(validity_real))

            z = torch.randn(real_data.shape[0], latent_dim).to(self.device)
            fake_data = self(z)
            validity_fake = self.discriminator(fake_data.detach())
            d_loss_fake = self.adversarial_loss(validity_fake, torch.zeros_like(validity_fake))

            d_loss = (d_loss_real + d_loss_fake) / 2
            self.log("d_loss", d_loss)
            return d_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return [opt_g, opt_d], []

# Step 5: Train the GAN
latent_dim = 100  # Size of the noise vector
data_dim = data_tensor.shape[1]  # Number of features in your dataset
lr = 0.0002

gan = GAN(latent_dim, data_dim, lr)
trainer = pl.Trainer(max_epochs=50, gpus=1)  # Use GPU
trainer.fit(gan, dataloader)

# Step 6: Generate Synthetic Data
num_samples = 17991  # Number of synthetic samples to generate
z = torch.randn(num_samples, latent_dim).to(gan.device)
synthetic_data = gan(z).detach().cpu().numpy()

# Step 7: Inverse Transform the Synthetic Data
synthetic_data = scaler.inverse_transform(synthetic_data)  # Rescale to original range

# Step 8: Save the Synthetic Data as a .csv File
synthetic_df = pd.DataFrame(synthetic_data, columns=data.columns)
synthetic_df.to_csv("synthetic_data.csv", index=False)

# Download the synthetic data
files.download("synthetic_data.csv")