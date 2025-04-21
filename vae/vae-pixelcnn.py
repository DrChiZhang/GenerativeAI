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
    """
    Safe YAML config loading from given path.

    Args:
        config_path (str): path to configuration YAML file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Error parsing YAML config:", exc)
            exit(1)  # Exit if YAML is malformed
    
def main():
    # --------------------- Parse arguments ---------------------
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c', dest="filename", metavar='FILE', help='Path to the config file', default='configs/vae.yaml')
    args = parser.parse_args()

    # --------------------- Load configuration ---------------------
    config = load_config(args.filename)

    # --------------------- Build model ---------------------
    # Ensure the requested model exists
    print(f"======= Setup and Load VQ-VAE =======")
    vae_model_name = config['model_params']['name']
    if vae_model_name not in vae_models:
        raise ValueError(f"Model '{vae_model_name}' not found in vae_models dict!")
    # Instantiate model using model_params from config
    vqvae = vae_models[config['model_params']['name']](**config['model_params'])

    ckpt = torch.load(config['model_params']['ckpt_path'])
    vqvae.load_state_dict(
        {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
    )
    vqvae.eval()
    vqvae = vqvae.cuda()
    # --------------------- Data module setup ---------------------
    num_gpus = config['trainer_params'].get('gpus', 0)
    pin_memory = bool(num_gpus)
    data = VAEDataset(**config["data_params"], pin_memory=pin_memory)
    data.setup()  # Prepare and split data
    train_loader = data.train_dataloader()

    # --------------------- Extract Codebook Indices on Full Dataset ---------------------
    all_codes = []
    all_attrs = []
    with torch.no_grad():
        for imgs, attrs in train_loader:
            imgs = imgs.to('cuda')
            codes = vqvae.get_codebook_indices(imgs).cpu()  # [B, H, W] (int64)
            all_codes.append(codes)
            all_attrs.append(attrs.float())                 # [B, n_classes], multi-hot attributes
    all_codes = torch.cat(all_codes, dim=0)                # [N, H, W]
    all_attrs = torch.cat(all_attrs, dim=0)                # [N, n_classes]
    flat_codes = all_codes.view(all_codes.shape[0], -1)    # [N, seq_len], seq_len = H * W

    # --------------------- PixelCNN training ---------------------
    # Prepare training tensor
    print(f"======= Training PixelCNN =======")
    ar_modle_name = config['ar_model_params']['name']
    if ar_modle_name not in ar_models:
        raise ValueError(f"AR model '{ar_modle_name}' not found!")
    grid_h, grid_w = all_codes.shape[1:3]
    num_tokens = vqvae.vq_layer.K
    n_classes = config['ar_model_params']['n_classes']
    embedding_dim = vqvae.vq_layer.embedding_dim
    latent_grid = (grid_h, grid_w)  # [H, W]
    pixelcnn = ar_models[config['ar_model_params']['name']](input_dim = num_tokens, 
                                                            dim = grid_h * grid_w, 
                                                            n_layers = 8, 
                                                            n_classes = n_classes
                                                            ).cuda()

    optimizer = torch.optim.Adam(pixelcnn.parameters(), lr=2e-4)

    n_epochs = 5  # for demo; increase for better results
    dataset = torch.utils.data.TensorDataset(flat_codes, all_attrs)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(n_epochs):
        for batch_codes, batch_attrs in loader:
            bsz = batch_codes.size(0)
            codes = batch_codes.view(bsz, *latent_grid).to('cuda')  # (B, H, W)
            attrs = batch_attrs.to('cuda')
            logits = pixelcnn(codes, attrs)
            loss = F.cross_entropy(logits.reshape(-1, num_tokens), codes.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch %d PixelCNN loss: %.4f" % (epoch+1, loss.item()))

    # --------------------- Sampling ---------------------
    # Sample From PixelCNN and Decode with VQ-VAE
    print(f"======= Sampling From PixelCNN and Decode with VQ-VAE =======")
    test_attrs = torch.zeros(10, n_classes)
    test_attrs[:, 20] = 1   # e.g. visualize with "Male" attribute ON (find correct index for your attribute)
    test_attrs = test_attrs.to('cuda')
    pixelcnn.eval()
    with torch.no_grad():
        sample_codes = pixelcnn.generate(test_attrs, shape=latent_grid, batch_size=10)
        # Map indices to codebook vectors and decode via VQ-VAE
        quantized = vqvae.vq_layer.embedding(sample_codes.view(-1)) \
                    .view(10, latent_grid[0], latent_grid[1], embedding_dim) \
                    .permute(0, 3, 1, 2).contiguous()
        decoded = vqvae.decoder(quantized).cpu()

    # --- Visualize generated images ---
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,2))
    for i in range(10):
        img_np = decoded[i].permute(1,2,0).clamp(0,1).numpy()
        plt.subplot(1,10,i+1)
        plt.imshow(img_np)
        plt.title(f'Sample {i}')
        plt.axis('off')
    plt.suptitle('VQ-VAE + PixelCNN (Male=ON)')
    plt.show()

if __name__ == "__main__":
    main()