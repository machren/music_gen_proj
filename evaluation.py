# evaluation.py

import os
import numpy as np
from scipy.linalg import sqrtm
import torch
from torchvision import models

def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def get_activations(loader, model, device):
    """
    Extract activations from a model for a given dataset.
    """
    activations = []
    for data in loader:
        x = data.to(device)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear')  # Resize for Inception
        act = model(x).detach().cpu().numpy()
    activations.append(act)

    return np.concatenate(activations, axis=0)

def get_activations_vae(vae, model, z_dim, n_samples, device):
    """
    Extract activations from a VAE-generated dataset.
    """
    activations = []
    for i in range(n_samples):
        z = torch.randn(1, z_dim).to(device)  # Generate random latent vector
        piano_roll = vae.decode(z)  # Decode latent vector to piano roll
        piano_roll = piano_roll.repeat(1, 3, 1, 1)  # Make RGB
        piano_roll = torch.nn.functional.interpolate(piano_roll, size=(299, 299), mode='bilinear')
        act = model(piano_roll).detach().cpu().numpy()  # Get activations
        activations.append(act)
    return np.concatenate(activations, axis=0)

def inception_score(p_yx, eps=1E-16):
    """
    The Inception Score, calculated as the exponential of the average KL divergence 
    between conditional and marginal probabilities.
    """
    # Compute the marginal class probabilities
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # Calculate KL divergence between conditional and marginal probabilities
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # Sum KL divergence over classes and average over images
    avg_kl_d = np.mean(kl_d.sum(axis=1))
    # Return the exponential of the average KL divergence
    return np.exp(avg_kl_d)

def get_predicted_probs(vae, inception_model, z_dim, n_samples, device):
    probs = []
    for _ in range(n_samples):
        z = torch.randn(1, z_dim).to(device)
        with torch.no_grad():
            piano_roll = vae.decode(z)
            piano_roll = piano_roll.repeat(1, 3, 1, 1)
            piano_roll = torch.nn.functional.interpolate(piano_roll, size=(299, 299), mode='bilinear')
            logits = inception_model(piano_roll)
            p = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
    return np.vstack(probs)
