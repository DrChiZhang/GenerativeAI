import os
import yaml
import argparse
from pathlib import Path

from models import *                  # vae_models dict should be defined here
from experiment import VAEXperiment   # Your experiment module
from dataset import VAEDataset        # Your datamodule implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

def load_config(config_path):
    """Safe YAML config loading from given path."""
    with open(config_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Error parsing YAML config:", exc)
            exit(1)

def main():
    # -------- Parse arguments --------
    parser = argparse.ArgumentParser(description='VAE+PixelCNN Conditional Generation Runner')
    parser.add_argument('--config', '-c', dest="filename", metavar='FILE', help='Path to YAML config file', default='configs/vae.yaml')
    args = parser.parse_args()

    # -------- Load config --------
    config = load_config(args.filename)

    # -------- Instantiate VQ-VAE model & load weights --------
    print(f"\n======= Setup and Load VQ-VAE =======")
    vae_model_name = config['vae_model_params']['name']
    assert vae_model_name in vae_models, f"Model '{vae_model_name}' not found in vae_models dict!"
    vqvae = vae_models[vae_model_name](**config['vae_model_params'])

    ckpt = torch.load(config['vae_model_params']['ckpt_path'], map_location='cpu')
    is_lightning = 'state_dict' in ckpt
    print("Loading checkpoint: Lightning-style state_dict =", is_lightning)
    try:
        if is_lightning:
            vqvae.load_state_dict(
                {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')},
                strict=False
            )
        else:
            vqvae.load_state_dict(ckpt, strict=False)
    except Exception as e:
        print(f"Failed to load state_dict: {e}")
        raise
    vqvae.eval().cuda()

    # -------- Setup Data ----------
    num_gpus = config['trainer_params'].get('gpus', 0)
    pin_memory = bool(num_gpus)
    data = VAEDataset(**config["data_params"], pin_memory=pin_memory)
    data.setup()
    train_loader = data.train_dataloader()

    # -------- Extract Codebook Indices and Attributes --------
    print("\n======= Extract Codebook Indices =======")
    all_codes, all_attrs = [], []
    with torch.no_grad():
        for imgs, attrs in train_loader:
            imgs = imgs.cuda()
            codes = vqvae.get_codebook_indices(imgs).cpu()
            all_codes.append(codes)
            all_attrs.append(attrs.float())
    all_codes = torch.cat(all_codes, dim=0)
    all_attrs = torch.cat(all_attrs, dim=0)
    flat_codes = all_codes.view(all_codes.shape[0], -1)
    print("All code indices shape:", all_codes.shape, "All attrs shape:", all_attrs.shape)

    # --------- Train PixelCNN Prior ---------
    print(f"\n======= Training PixelCNN =======")
    ar_model_name = config['ar_model_params']['name']
    assert ar_model_name in ar_models, f"AR model '{ar_model_name}' not found!"
    grid_h, grid_w = all_codes.shape[1:3]
    num_tokens = vqvae.vq_layer.K
    n_classes = config['ar_model_params']['n_classes']
    embedding_dim = vqvae.vq_layer.D
    latent_grid = (grid_h, grid_w)

    # PixelCNN channel dim should be moderate (e.g. 64, not grid_h * grid_w)
    pcnn_dim = config['ar_model_params'].get('dim', 64)
    pixelcnn = ar_models[ar_model_name](
        input_dim=num_tokens,
        dim=pcnn_dim,
        n_layers=config['ar_model_params'].get('n_layers', 8),
        n_classes=n_classes
    ).cuda()
    optimizer = torch.optim.Adam(pixelcnn.parameters(), lr=2e-4)
    dataset = torch.utils.data.TensorDataset(flat_codes, all_attrs)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    n_epochs = config['ar_model_params'].get('ar_epochs', 5)
    for epoch in range(n_epochs):
        for batch_codes, batch_attrs in loader:
            bsz = batch_codes.size(0)
            codes = batch_codes.view(bsz, *latent_grid).cuda()
            attrs = batch_attrs.cuda()
            logits = pixelcnn(codes, attrs)
            loss = F.cross_entropy(logits.reshape(-1, num_tokens), codes.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} PixelCNN loss: {loss.item():.4f}")

    # --------- Sampling from Prior & VQ-VAE Decode ---------
    print(f"\n======= Sampling From PixelCNN and Decode with VQ-VAE =======")
    # Attribute index 20 for 'Male', check your attribute ordering!
    test_attrs = torch.zeros(10, n_classes)
    test_attrs[:, 20] = 1   # Each sample with only 'Male' ON
    test_attrs = test_attrs.cuda()
    pixelcnn.eval()
    with torch.no_grad():
        # Generate attribute-conditional VQ code grids
        sample_codes = pixelcnn.generate(test_attrs, shape=latent_grid, batch_size=10)
        quantized = vqvae.vq_layer.embedding(sample_codes.view(-1))
        quantized = quantized.view(10, latent_grid[0], latent_grid[1], embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        decoded = vqvae.decoder(quantized).cpu()

    # --------- Visualize ---------
    print(f"\n======= Visualized Generated Image =======")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 2))
    for i in range(10):
        img_np = decoded[i].permute(1,2,0).clamp(0,1).numpy()
        plt.subplot(1, 10, i+1)
        plt.imshow(img_np)
        plt.title(f'Sample {i}')
        plt.axis('off')
    plt.suptitle('VQ-VAE + PixelCNN (Male=ON)')
    plt.show()

if __name__ == "__main__":
    main()