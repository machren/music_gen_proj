# train_vae_gan.py

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import config

def vae_loss(x_hat, x, mean, logvar):
    BCE = F.mse_loss(x_hat, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
    return BCE + KLD, BCE, KLD


def discriminator_loss(real_logits, fake_logits):
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return real_loss + fake_loss


def generator_adv_loss(fake_logits):
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def train_vae_gan(vae, discriminator, train_loader, n_epochs, device, lambda_adv=1.0):
    vae = vae.to(device)
    discriminator = discriminator.to(device)

    optimizer_vae = optim.Adam(vae.parameters(), lr=config.LR)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.LR)

    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, 'vae_gan'))
    global_step = 0

    for epoch in range(n_epochs):
        total_loss = 0
        print(f"\n[Epoch {epoch + 1}/{n_epochs}]")
        for i, x in enumerate(train_loader):
            x = x.to(device)

            # === 1. Train Discriminator ===
            with torch.no_grad():
                x_hat, _, _ = vae(x)
                if x_hat.shape != x.shape:
                    x_hat = F.interpolate(x_hat, size=x.shape[2:], mode='bilinear', align_corners=False)

            real_logits = discriminator(x)
            fake_logits = discriminator(x_hat.detach())
            d_loss = discriminator_loss(real_logits, fake_logits)
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # === 2. Train VAE (decoder acts as generator) ===
            x_hat, mean, logvar = vae(x)
            if x_hat.shape != x.shape:
                x_hat = F.interpolate(x_hat, size=x.shape[2:], mode='bilinear', align_corners=False)

            fake_logits = discriminator(x_hat)
            total_vae_loss, recon_loss, kld_loss = vae_loss(x_hat, x, mean, logvar)
            adv_loss = generator_adv_loss(fake_logits)
            total_loss_batch = total_vae_loss + lambda_adv * adv_loss

            optimizer_vae.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer_vae.step()

            writer.add_scalar("VAE_GAN/Total_Loss", total_loss_batch.item(), global_step)
            writer.add_scalar("VAE_GAN/Reconstruction_Loss", recon_loss.item(), global_step)
            writer.add_scalar("VAE_GAN/KLD_Loss", kld_loss.item(), global_step)
            writer.add_scalar("VAE_GAN/Adversarial_Loss", adv_loss.item(), global_step)
            global_step += 1

            if i % 50 == 0:
                print(f"  [Batch {i}]  Total: {total_loss_batch.item():.4f}  Recon: {recon_loss.item():.4f}  KLD: {kld_loss.item():.4f}  Adv: {adv_loss.item():.4f}")

            total_loss += total_loss_batch.item()

        print(f"Epoch {epoch+1} complete — Avg Loss: {total_loss / len(train_loader):.4f}")

    writer.close()

    os.makedirs(config.MOD_DIR, exist_ok=True)
    torch.save(vae.state_dict(), os.path.join(config.MOD_DIR, 'vae_gan_decoder.pth'))
    torch.save(discriminator.state_dict(), os.path.join(config.MOD_DIR, 'vae_gan_discriminator.pth'))
    print("VAE-GAN models saved.")

    return vae, discriminator

