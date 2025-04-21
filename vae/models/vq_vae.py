import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Union, Tuple

from .types_ import *

class VectorQuantizer(nn.Module):
    """
    Implements the vector quantization layer of VQ-VAE.
    Discretizes continuous encoder outputs using a shared codebook.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super().__init__()
        self.K = num_embeddings      # Number of codebook vectors
        self.D = embedding_dim       # Dimensionality of codebook vectors
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Discretizes each D-dimensional latent in the input feature map
        to the nearest entry in the codebook, then returns quantized output and loss term.
        Args:
            latents: (B, D, H, W) - latent feature map
        Returns:
            quantized_latents: (B, D, H, W) - quantized using codebook
            vq_loss:            - loss term to add to recon loss
        """
        # Permute to (B, H, W, D) so each D-vector is a code to quantize
        latents = latents.permute(0, 2, 3, 1).contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # (B*H*W, D)

        # L2 distances from each continuous latent to all codebook vectors
        dist = (
            torch.sum(flat_latents ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )  # (B*H*W, K)

        # Find nearest embedding for each latent
        encoding_inds = torch.argmin(dist, dim=1, keepdim=True)  # (B*H*W, 1)

        # One-hot encode these indices
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)

        # Quantize: multiply one-hot with embedding weights
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # (B*H*W, D)
        quantized_latents = quantized_latents.view(latents_shape)  # (B, H, W, D)

        # VQ-VAE Losses (commitment + codebook)
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        vq_loss = self.beta * commitment_loss + embedding_loss

        # Straight-through estimator for backward
        quantized_latents = latents + (quantized_latents - latents).detach()

        # Return to (B, D, H, W)
        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss

class ResidualLayer(nn.Module):
    """A residual block for improved expressivity."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)

class VQVAE(nn.Module):
    """
    Full VQ-VAE model: Encoder, VectorQuantizer, Decoder
    """
    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List[int] = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 **kwargs) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # --- Encoder ---
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())
        )

        # Add 6 residual blocks
        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        # Final embedding projection
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)

        # --- Decoder ---
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims = hidden_dims[::-1]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                      kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], out_channels=3, kernel_size=4, stride=2, padding=1),
                nn.Tanh())
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encode image to continuous latent codes.
        :param input: (N, C, H, W)
        :return: [latent tensor]
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode quantized codebook output (B, D, H, W) back to image.
        """
        return self.decoder(z)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Full VQ-VAE forward: encode, quantize, decode.
        """
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self, *args, **kwargs) -> VAEOutput:
        """
        Compute loss for VQ-VAE:
        - MSE/Recon loss
        - VQ (embedding, commitment) loss
        """
        recons, input, vq_loss = args[0], args[1], args[2]
        assert torch.isfinite(recons).all(), "reconstruction output contains inf/nan"
        assert torch.isfinite(input).all(), "input image contains inf/nan"
        assert torch.isfinite(vq_loss).all(), "vq_loss is inf/nan"

        recons_loss = F.mse_loss(recons, input)
        loss = recons_loss + vq_loss
        return VAEOutput(loss=loss, recon_loss=recons_loss.detach(), vq_loss=vq_loss.detach())

    def sample(self, num_samples: int, current_device: Union[int, str], **kwargs) -> Tensor:
        raise NotImplementedError("VQVAE sampling requires a prior over indices (e.g., PixelCNN).")

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Reconstruct input images through codebook.
        """
        return self.forward(x)[0]

    def get_codebook_indices(self, x: Tensor) -> Tensor:
        """
        Utility to extract codebook indices for a batch of images -
        useful for training priors (PixelCNN) or visualization.
        """
        encoding = self.encode(x)[0]
        latents = encoding.permute(0, 2, 3, 1).contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.embedding_dim)
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.vq_layer.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.vq_layer.embedding.weight.t())
        encoding_inds = torch.argmin(dist, dim=1)
        return encoding_inds.view(latents_shape[0], latents_shape[1], latents_shape[2])

    # Optionally: add visualization, codebook hooks, etc.