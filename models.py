# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim, n_features, n_channels):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, n_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_features * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_features * 8, n_features * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_features * 4, n_features * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_features * 2, n_features, (5, 4), (3, 2), (1, 1), bias=False),
            nn.BatchNorm2d(n_features),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_features, n_channels, (6, 4), (2, 4), (1, 1), bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.gen(input)


class Discriminator(nn.Module):
    def __init__(self, n_channels, n_features):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, n_features, (6, 4), (2, 4), (1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features, n_features * 2, (5, 4), (3, 2), (1, 1), bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features * 2, n_features * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features * 4, n_features * 8, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(n_features * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),             
            nn.Linear(n_features * 8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(-1)

        
class VAE(nn.Module):
    def __init__(self, n_channels, n_features, z_dim):
        super(VAE, self).__init__()

        # --- Encoder ---
        self.enc_conv1 = nn.Conv2d(n_channels, n_features, kernel_size=(6, 4), stride=(2, 4), padding=(1, 1))
        self.enc_conv2 = nn.Conv2d(n_features, n_features*2, kernel_size=(5, 4), stride=(3, 2), padding=(1, 1))
        self.enc_conv3 = nn.Conv2d(n_features*2, n_features*4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.enc_conv4 = nn.Conv2d(n_features*4, n_features*8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.enc_mean = nn.Linear(n_features*8*4*2, z_dim)
        self.enc_logvar = nn.Linear(n_features*8*4*2, z_dim)
        
        # --- Decoder ---
        self.dec_lin1 = nn.Linear(z_dim, n_features*8*4*2)
        self.dec_conv1 = nn.ConvTranspose2d(n_features*8, n_features*4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.dec_conv2 = nn.ConvTranspose2d(n_features*4, n_features*2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.dec_conv3 = nn.ConvTranspose2d(n_features*2, n_features, kernel_size=(5, 4), stride=(3, 2), padding=(1, 1))
        self.dec_conv4 = nn.ConvTranspose2d(n_features, n_channels, kernel_size=(6, 4), stride=(2, 4), padding=(1, 1))

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = x.view(x.size(0), -1)
        mean = self.enc_mean(x)
        logvar = self.enc_logvar(x)
        logvar = torch.clamp(logvar, min=-10, max=10) # To prevent loss explotion
        return mean, logvar

    def decode(self, z):
        x = F.relu(self.dec_lin1(z))
        x = x.view(x.size(0), -1, 4, 2)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = self.dec_conv4(x)
        return x

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
