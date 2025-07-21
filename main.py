# main.py

import os
import glob
import pandas as pd
import torch

import config
import data_preprocessing
import models
import train
from train import train_gan
import evaluation
import visualization

def main():
    # --- 1. Data Preprocessing ---
    if config.MAKE_DATASET_DIR:
        print("Starting data preprocessing...")
        filenames = glob.glob(os.path.join(config.GIVEN_DATA_DIR, '**/*.mid*'), recursive=True)
        metadata = pd.read_csv(config.METADATA_FILE)
        data_preprocessing.prepare_dataset(
            filenames,
            metadata,
            config.DATASET_DIR,
            make_dir=config.MAKE_DATASET_DIR,
            fs=config.FS,
            pitch_range=config.PITCH_RANGE,
            chunk_size=config.CHUNK_SIZE
        )
        print("Data preprocessing complete.")

    train_loader, test_loader, _ = data_preprocessing.get_dataloaders()

    # --- 2. Initialize Models ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = models.Generator(config.Z_DIM, config.N_FEATURES, config.N_CHANNELS).to(device)
    netD = models.Discriminator(config.N_CHANNELS, config.N_FEATURES).to(device)
    vae = models.VAE(config.N_CHANNELS, config.N_FEATURES_VAE, config.Z_DIM_VAE).to(device)

    # --- 3. Train GAN ---
    print("\n--- Training GAN ---")
    D_losses, G_losses, img_list = train.train_gan(
        netD, netG, train_loader, config.N_EPOCHS, config.Z_DIM, device
    )
    visualization.plot_losses(D_losses, G_losses)
    if not os.path.exists(config.GENERATED_IMAGES_DIR):
        os.makedirs(config.GENERATED_IMAGES_DIR)
    visualization.save_images(img_list, config.GENERATED_IMAGES_DIR)
    print("GAN training complete. Images saved.")

    # --- 4. Train VAE ---
    print("\n--- Training VAE ---")
    losses_vae = train.train_vae(vae, train_loader, config.N_EPOCHS, device)
    # visualization.plot_losses_vae(losses_vae) # You can create this function in visualization.py
    print("VAE training complete.")

    # --- 5. Evaluation (Optional) ---
    print("\n--- Evaluating Models ---")
    # You can add your evaluation logic here, calling functions from evaluation.py


if __name__ == '__main__':
    main()