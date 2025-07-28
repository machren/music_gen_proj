import os
import glob
import pandas as pd
import torch
import torchvision.models as models_tv

import config
import data_preprocessing
import models
import train
from train import train_gan
from train_vae_gan import train_vae_gan
import evaluation
import visualization
import generation


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

    train_loader, test_loader, _ = data_preprocessing.get_dataloaders(fraction=config.FRACTION)

    # --- 2. Initialize Models ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = models.Generator(config.Z_DIM, config.N_FEATURES, config.N_CHANNELS).to(device)
    netD = models.Discriminator(config.N_CHANNELS, config.N_FEATURES).to(device)
    vae = models.VAE(config.N_CHANNELS, config.N_FEATURES_VAE, config.Z_DIM_VAE).to(device)

    train_mode = config.TRAIN_MODE.lower()
    print(f"\nSelected mode: {train_mode}")

    if train_mode == "gan":
        # --- 3. Train GAN ---
        print("\n--- Training GAN ---")
        D_losses, G_losses, img_list = train.train_gan(
            netD, netG, train_loader, config.N_EPOCHS, config.Z_DIM, device
        )
        visualization.plot_losses(D_losses, G_losses)
        save_dir = os.path.join(config.GENERATED_IMAGES_DIR, 'gan')
        os.makedirs(save_dir, exist_ok=True)
        visualization.save_images(img_list, save_dir)
        print("GAN training complete. Images saved.")

        # --- Evaluation and Generation for GAN ---

        evaluate_and_generate(netG, model_type="gan", z_dim=config.Z_DIM_VAE, device=device, train_loader=train_loader)
    
    elif train_mode == "vae":
        # --- 4. Train VAE ---
        print("\n--- Training VAE ---")
        train.train_vae(vae, train_loader, config.N_EPOCHS, device)
        print("VAE training complete.")

        evaluate_and_generate(vae, model_type="vae", z_dim=config.Z_DIM_VAE, device=device, train_loader=train_loader)

    elif train_mode == "vae_gan":
        # --- 5. Train VAE-GAN ---
        print("\n--- Training VAE-GAN ---")
        vae, disc = train_vae_gan(vae, netD, train_loader, config.N_EPOCHS, device)
        print("VAE-GAN training complete.")

        evaluate_and_generate(vae, model_type="vae_gan", z_dim=config.Z_DIM, device=device, train_loader=train_loader)

    else:
        raise ValueError(f"Unsupported TRAIN_MODE: {train_mode}")


def evaluate_and_generate(vae_model, model_type, z_dim, device, train_loader):
    print("\n--- Evaluating and Generating ---")

    # 1. Evaluation
    inception_model = models_tv.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    print("Calculating real activations...")
    real_activations = evaluation.get_activations(train_loader, inception_model, device)

    print("Calculating generated activations...")
    gen_activations = evaluation.get_activations_generic(
        vae_model, inception_model, z_dim, config.N_SAMPLES, device, model_type
    )

    fid = evaluation.calculate_fid(real_activations, gen_activations)
    print(f"FID score: {fid:.4f}")

    p_yx = evaluation.get_predicted_probs(vae_model, inception_model, z_dim, config.N_SAMPLES, device, model_type)
    is_score = evaluation.inception_score(p_yx)
    print(f"Inception Score: {is_score:.4f}")

    # 2. Generation
    print("\nGenerating piano rolls and converting to MIDI...")
    gen_roll_dir = os.path.join(config.GENERATED_ROLLS_DIR, model_type)
    gen_midi_dir = os.path.join(config.GENERATED_MIDI_DIR, model_type)

    generation.save_generated_piano_rolls(vae_model, z_dim, config.N_SAMPLES, device, gen_roll_dir, model_type)
    generation.convert_saved_piano_rolls_to_midis(gen_roll_dir, gen_midi_dir, config.FS)

    print(f"Generated piano rolls saved to: {gen_roll_dir}")
    print(f"Generated MIDI files saved to: {gen_midi_dir}")


if __name__ == '__main__':
    main()
