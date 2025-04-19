import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Any, Dict
from .types_ import *


class ConditionalVAE(nn.Module):  # Now inherit from torch.nn.Module
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_dim: int,
                 hidden_dims: List[int] = None,
                 img_size: int = 64,
                 recon_loss_type: str = "mse",        # "mse" or "bce"
                 **kwargs) -> None:
        super().__init__()  # Call nn.Module's constructor
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_classes = num_classes
        self.recon_loss_type = recon_loss_type.lower()

        # Layer to embed the label information into a spatial map
        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Build the encoder network
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims.copy()   # Store for decoder

        encoder_in_channels = in_channels + 1  # Account for the extra label channel
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(encoder_in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU())
            )
            encoder_in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Compute feature size after encoder for the dense layers
        self.encoder_output_size = (
            hidden_dims[-1],
            img_size // (2 ** len(hidden_dims)),
            img_size // (2 ** len(hidden_dims))
        )
        feat_dim = (
            self.encoder_output_size[0] *
            self.encoder_output_size[1] *
            self.encoder_output_size[2]
        )

        # Map flat encoder output to latent mean and variance
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_var = nn.Linear(feat_dim, latent_dim)

        # Build the decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim + num_classes, feat_dim)
        hidden_dims = hidden_dims[::-1]  # Reverse for decoder
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        # Final layer brings number of channels and value range back to image
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encode(self, x: Tensor) -> List[Tensor]:
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(z.size(0), *self.encoder_output_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, 
                input: Tensor, 
                labels: Tensor, 
                **kwargs
        ) -> List[Tensor]:
        """
        Forward pass for the Conditional VAE.
        """
        y = labels.float()
        embedded_class = self.embed_class(y).view(-1, 1, self.img_size, self.img_size)
        embedded_input = self.embed_data(input)
        x = torch.cat([embedded_input, embedded_class], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, y], dim=1)
        return [self.decode(z), input, mu, log_var]

    def loss_function(
        self, 
        *args: Any,
        **kwargs: Any
    ) -> VAEOutput:
        """
        Computes loss for VAE.
        Args:
            recons: Reconstructed
            input: Original
            mu, log_var: Latent mean & logvar
            kld_weight: scaling for KL
        Returns:
            VAEOutput : dataclass with loss, recon_loss, kld
        """
        recons, input, mu, log_var = args[0], args[1], args[2], args[3] # Unpack results
        kld_weight = kwargs['kld_weight']  # Account for the minibatch samples from the dataset

        if self.recon_loss_type == 'bce':
            recon_loss = F.binary_cross_entropy(recons, input, reduction='mean')
        else:
            recon_loss = F.mse_loss(recons, input)

        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        loss = recon_loss + kld_weight * kld
        return VAEOutput(loss=loss, recon_loss=recon_loss.detach(), kld_loss=kld.detach())

    def sample(self, 
               batch_size: int, 
               device: torch.device, 
               labels: Tensor
    ) -> Tensor:
        """
        Generate random samples from the VAE's latent space.
        """
        y = labels.float().to(device)
        z = torch.randn(batch_size, self.latent_dim, device=device)
        print(f"z.shape: {z.shape}, y.shape: {y.shape}")
        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, 
                 x: Tensor, 
                 labels: Tensor, 
                 **kwargs) -> Tensor:
        return self.forward(x, labels=labels)[0]