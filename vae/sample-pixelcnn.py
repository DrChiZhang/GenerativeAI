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

class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, grid_size, hidden=128, n_layers=7):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, hidden)
        self.layers = nn.ModuleList([
            nn.Conv2d(hidden, hidden, 7, padding=3) for _ in range(n_layers)
        ])
        self.out = nn.Conv2d(hidden, num_embeddings, 1)
    def forward(self, x):
        # x: (batch, H, W), torch.long
        x = self.emb(x)                     # (batch, H, W, hidden)
        x = x.permute(0,3,1,2)              # (batch, hidden, H, W)
        for l in self.layers:
            x = F.relu(l(x))
        return self.out(x)                  # logits, (batch, num_embeds, H, W)
    
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
    model_name = config['model_params']['name']
    if model_name not in vae_models:
        raise ValueError(f"Model '{model_name}' not found in vae_models dict!")
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
    vqvae.eval()
    with torch.no_grad():
        for imgs, _ in train_loader :  # Use the train data loader when extracting codebook indices for training the prior.
            imgs = imgs.to('cuda')
            codes = vqvae.get_codebook_indices(imgs).cpu()  # [B, H, W], dtype torch.long
            all_codes.append(codes)
    all_codes = torch.cat(all_codes, 0)  # shape: [N, H, W]
    print(all_codes.shape)

    # --------------------- PixelCNN training ---------------------
    # Prepare training tensor
    print(f"======= Training PixelCNN =======")
    grid_h, grid_w = all_codes.shape[1:3]
    num_tokens = vqvae.vq_layer.K
    pixelcnn = PixelCNN(num_tokens, grid_size=(grid_h, grid_w), hidden=128).cuda()

    optimizer = torch.optim.Adam(pixelcnn.parameters(), lr=2e-4)
    batch_size = 64
    for epoch in range(5):
        perm = torch.randperm(all_codes.shape[0])
        for i in range(0, all_codes.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            codes = all_codes[idx].to('cuda') # shape: (B, H, W)
            logits = pixelcnn(codes)
            # Reshape and train as classification: next-token prediction or standard pixelwise cross-entropy
            loss = F.cross_entropy(
                logits.contiguous().view(-1, num_tokens),
                codes.contiguous().view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"PixelCNN epoch {epoch+1} loss: {loss.item():.4f}")

    # --------------------- Sampling ---------------------
    # Sample From PixelCNN and Decode with VQ-VAE
    print(f"======= Sampling From PixelCNN and Decode with VQ-VAE =======")
    pixelcnn.eval()
    codes = torch.zeros(1, grid_h, grid_w, dtype=torch.long, device='cuda')
    with torch.no_grad():
        for i in range(grid_h):
            for j in range(grid_w):
                logits = pixelcnn(codes)
                probs = torch.softmax(logits[0, :, i, j], -1)
                sampled = torch.multinomial(probs, 1)
                codes[0, i, j] = sampled

    # Convert indices into codebook vectors for VQ-VAE decoder
    quantized = vqvae.vq_layer.embedding(codes.view(-1))      # (H*W, D)
    quantized = quantized.view(1, grid_h, grid_w, vqvae.embedding_dim)
    quantized = quantized.permute(0, 3, 1, 2).contiguous()    # (1, D, H, W)
    img_pred = vqvae.decoder(quantized)                       # (1, 3, H_img, W_img)

    import matplotlib.pyplot as plt
    plt.imshow(img_pred[0].cpu().permute(1,2,0).clamp(0,1).detach().numpy())
    plt.axis('off')
    plt.title("VQ-VAE + PixelCNN sample")
    plt.show()

if __name__ == "__main__":
    main()