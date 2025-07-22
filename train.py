# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
import os

import config

def train_gan(netD, netG, train_loader, n_epochs, z_dim, device):
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, 'gan'))

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

    global_step = 0

    for epoch in range(n_epochs):
        print(f"\n[Epoch {epoch + 1}/{n_epochs}]")
        for i, data in enumerate(train_loader, 0):
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

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            D_total = errD_real.item() + errD_fake.item()
            G_losses.append(errG.item())
            D_losses.append(D_total)

            # Log per iteration
            writer.add_scalar("GAN/Discriminator Loss", D_total, global_step)
            writer.add_scalar("GAN/Generator Loss", errG.item(), global_step)
            global_step += 1

            if i % 50 == 0:
                print(f"  [Batch {i}/{len(train_loader)}]  D_loss: {D_total:.4f}  G_loss: {errG.item():.4f}")

            if (i % 500 == 0) or ((epoch == n_epochs-1) and (i == len(train_loader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_grid = (fake[:16] + 1) / 2  # Rescale [-1,1] → [0,1]
                writer.add_images("GAN/Samples", img_grid, global_step)

        writer.add_scalar("GAN/Avg D Loss (epoch)", sum(D_losses[-len(train_loader):]) / len(train_loader), epoch)
        writer.add_scalar("GAN/Avg G Loss (epoch)", sum(G_losses[-len(train_loader):]) / len(train_loader), epoch)

    writer.close()
    return D_losses, G_losses, img_list

def vae_loss(x, x_hat, mean, logvar):
    bce = nn.BCELoss(reduction='sum')
    BCE = bce(x_hat, x)
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(vae, train_loader, n_epochs, device):
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, 'vae'))
    optimizer_vae = optim.Adam(vae.parameters(), lr=config.LR)
    losses = []

    global_step = 0

    for epoch in range(n_epochs):
        epoch_loss = 0
        print(f"\n[Epoch {epoch + 1}/{n_epochs}]")
        for i, data in enumerate(train_loader, 0):
            x = data.to(device)
            optimizer_vae.zero_grad()
            x_hat, mean, logvar = vae(x)
            loss = vae_loss(x, x_hat, mean, logvar)
            loss.backward()
            optimizer_vae.step()

            losses.append(loss.item())
            epoch_loss += loss.item()

            writer.add_scalar("VAE/Loss", loss.item(), global_step)
            global_step += 1

            if i % 50 == 0:
                print(f"  [Batch {i}/{len(train_loader)}]  Batch Loss: {loss.item() / len(data):.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        writer.add_scalar("VAE/Avg Epoch Loss", avg_epoch_loss, epoch)
        print(f"Epoch {epoch + 1} complete — Avg Loss: {avg_epoch_loss:.4f}")

    writer.close()
    return losses
