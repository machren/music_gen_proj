# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import config

def train_gan(netD, netG, train_loader, n_epochs, z_dim, device):  
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))

    D_losses = []
    G_losses = []
    img_list = []
    real_label = 1.
    fake_label = 0.
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    netD = netD.float()
    netG = netG.float()

    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader, 0):
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            real_cpu = data[0].to(device) if isinstance(data, list) else data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()

            # Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            # Save losses for analysis
            G_losses.append(errG.item())
            D_losses.append(errD_real.item() + errD_fake.item())

            # Save generator output periodically
            if (i % 500 == 0) or ((epoch == n_epochs-1) and (i == len(train_loader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                normalized_fake = (fake[0] - fake[0].min()) / (fake[0].max() - fake[0].min())
                img_list.append(transforms.ToPILImage()(normalized_fake))
    return D_losses, G_losses, img_list

def vae_loss(x, x_hat, mean, logvar):
    bce = nn.BCELoss(reduction='sum')
    BCE = bce(x_hat, x)
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(vae, train_loader, n_epochs, device):
    optimizer_vae = optim.Adam(vae.parameters(), lr=config.LR)
    losses = []
    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader, 0):
            # Load data and move to device
            x = data.to(device)
            
            # Reset gradients
            optimizer_vae.zero_grad()
            
            # Forward pass through VAE
            x_hat, mean, logvar = vae(x)
            
            # Compute loss
            loss = vae_loss(x, x_hat, mean, logvar)
            
            # Backpropagate and update weights
            loss.backward()
            optimizer_vae.step()
            
            # Record loss
            losses.append(loss.item())
            
            # Print progress periodically
            if i % 100 == 0:
                print(f'Epoch: {epoch+1}/{n_epochs}, Loss: {loss.item()/len(data):.4f}')
            pass
    return losses