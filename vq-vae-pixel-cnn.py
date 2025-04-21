import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np

# ---- VQ-VAE code (use your refined implementation here) ----

@dataclass
class VAEOutput:
    loss: torch.Tensor
    recon_loss: torch.Tensor
    vq_loss: torch.Tensor

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
    def forward(self, latents: torch.Tensor):
        latents = latents.permute(0,2,3,1).contiguous()
        shape = latents.shape
        flat_latents = latents.view(-1, self.D)
        dist = (
            torch.sum(flat_latents**2,dim=1,keepdim=True)
            + torch.sum(self.embedding.weight**2,dim=1)
            - 2*torch.matmul(flat_latents, self.embedding.weight.t())
        )
        encoding_inds = torch.argmin(dist,dim=1,keepdim=True)
        one_hot = torch.zeros(encoding_inds.size(0), self.K, device=latents.device)
        one_hot.scatter_(1, encoding_inds, 1)
        quantized = torch.matmul(one_hot, self.embedding.weight)
        quantized = quantized.view(shape)
        commitment_loss = F.mse_loss(quantized.detach(), latents)
        embedding_loss = F.mse_loss(quantized, latents.detach())
        vq_loss = self.beta * commitment_loss + embedding_loss
        quantized = latents + (quantized - latents).detach()
        return quantized.permute(0,3,1,2).contiguous(), vq_loss

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
        )
    def forward(self, x): return x + self.resblock(x)

class VQVAE(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_embeddings, hidden_dims=None, beta=0.25):
        super().__init__()
        modules = []
        if hidden_dims is None: hidden_dims=[64,128]
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, 4, 2, 1),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        for _ in range(2):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.Conv2d(in_channels, embedding_dim, 1))
        self.encoder = nn.Sequential(*modules)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, beta)
        dec_mod = []
        dec_mod.append(nn.Conv2d(embedding_dim, hidden_dims[-1], 3, 1, 1))
        for _ in range(2):
            dec_mod.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
        dec_mod.append(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[0], 4, 2, 1))
        dec_mod.append(nn.LeakyReLU())
        dec_mod.append(nn.ConvTranspose2d(hidden_dims[0], 1, 4, 2, 1))
        dec_mod.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_mod)
    def forward(self, x):
        z = self.encoder(x)
        q, vq_loss = self.vq(z)
        out = self.decoder(q)
        return out, x, vq_loss
    def loss_function(self, *args):
        recons, input, vq_loss = args
        recons_loss = F.mse_loss(recons, input)
        loss = recons_loss + vq_loss
        return VAEOutput(loss, recons_loss.detach(), vq_loss.detach())
    def get_codebook_indices(self, x):
        z = self.encoder(x)
        latents = z.permute(0,2,3,1).contiguous()
        flat_latents = latents.view(-1, self.vq.D)
        dist = (
            torch.sum(flat_latents**2, dim=1,keepdim=True)
            + torch.sum(self.vq.embedding.weight**2, dim=1)
            - 2*torch.matmul(flat_latents, self.vq.embedding.weight.t())
        )
        inds = torch.argmin(dist,dim=1)
        N, H, W, D = latents.shape
        return inds.view(N, H, W)

# ---- Minimal PixelCNN ----

class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, grid_size, hidden=64, n_layers=7):
        super().__init__()
        self.grid_size = grid_size
        self.emb = nn.Embedding(num_embeddings, hidden)
        self.layers = nn.ModuleList([
            nn.Conv2d(hidden, hidden, 3, padding=1) for _ in range(n_layers)
        ])
        self.out = nn.Conv2d(hidden, num_embeddings, 1)
    def forward(self, x):
        # x: (batch, H, W), torch.long
        x = self.emb(x)                     # (batch, H, W, hidden)
        x = x.permute(0,3,1,2)              # (batch, hidden, H, W)
        for l in self.layers:
            x = F.relu(l(x))
        return self.out(x)                  # logits, shape (batch, num_embeds, H, W)

# ------ TRAIN VQ-VAE AND PIXELCNN ON MNIST ------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vqvae = VQVAE(1, 8, 64, [64,128], beta=0.25).to(device)

trainset = torchvision.datasets.MNIST('./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
loader = DataLoader(trainset, batch_size=64, shuffle=True)

print("Training VQ-VAE...")
vqvae.train()
opt = torch.optim.Adam(vqvae.parameters(), lr=2e-3)
for epoch in range(4):
    for img, _ in loader:
        img = img.to(device)
        out = vqvae(img)
        vo = vqvae.loss_function(*out)
        opt.zero_grad()
        vo.loss.backward()
        opt.step()
    print(f'VQ-VAE Epoch {epoch+1} Loss: {vo.loss.item():.4f}')

# ------ Extract codebook indices ------
print("Extracting codebook indices...")
vqvae.eval()
all_codes = []
with torch.no_grad():
    for img, _ in loader:
        img = img.to(device)
        code = vqvae.get_codebook_indices(img).cpu()
        all_codes.append(code)
all_codes = torch.cat(all_codes, dim=0) # Shape: (N, H, W)
grid_size = all_codes.shape[1]

# ------ Train a simple PixelCNN ------
print("Training PixelCNN prior on codes...")
pixelcnn = PixelCNN(64, grid_size).to(device)
prior_opt = torch.optim.Adam(pixelcnn.parameters(), lr=2e-3)
batch_size = 16
for epoch in range(4):
    perm = torch.randperm(all_codes.shape[0])
    for i in range(0, all_codes.shape[0], batch_size):
        idx = perm[i:i+batch_size]
        codes = all_codes[idx].to(device)  # (B, H, W)
        logits = pixelcnn(codes[:, :-1, :-1])        # Predict grid excluding last row/col
        target = codes[:, 1:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, 64), target.reshape(-1))
        prior_opt.zero_grad()
        loss.backward()
        prior_opt.step()
    print(f'PixelCNN Epoch {epoch+1} Loss: {loss.item():.4f}')

# ------ SAMPLE FROM PIXELCNN & GENERATE NEW IMAGES ------
print("Generating images from PixelCNN prior...")
pixelcnn.eval()
samples = torch.zeros(1, grid_size, grid_size, dtype=torch.long, device=device)
with torch.no_grad():
    for i in range(grid_size):
        for j in range(grid_size):
            logits = pixelcnn(samples)
            probs = F.softmax(logits[0, :, i, j], dim=-1)
            samples[0, i, j] = torch.multinomial(probs, 1)
    # Now samples: (1, H, W), values are code indices

# Look up embeddings and decode to image
emb = vqvae.vq.embedding(samples.view(-1)).view(1, grid_size, grid_size, vqvae.vq.D)
emb = emb.permute(0,3,1,2).contiguous()
img_gen = vqvae.decoder(emb).cpu().detach().squeeze().numpy()

plt.title("VQ-VAE + PixelCNN Generated Sample")
plt.imshow(img_gen, cmap='gray')
plt.axis('off')
plt.show()