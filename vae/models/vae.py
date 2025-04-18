import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Callable, Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class VAEOutput:
    loss: torch.Tensor
    recon_loss: torch.Tensor
    kld: torch.Tensor

class VAE(nn.Module):
    def __init__(
        self,
        x_dim: int,
        input_shape: Tuple[int, int, int],   # (C, H, W)
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 16,
        activation: Optional[Callable[[], nn.Module]] = None,
        recon_loss_type: str = "mse",        # "mse" or "bce"
        **kwargs: Any
    ) -> None:
        """
        Args:
            x_dim : int : Number of channels in input image
            input_shape : (C,H,W) : Input image shape (tuple)
            hidden_dims : List[int] : List of channels for hidden layers
            latent_dim : int : Dimensionality of latent space
            activation : Callable : Activation (nn.ReLU, nn.LeakyReLU, etc.)
            recon_loss_type : str : Reconstruction loss ("mse" or "bce")
        """
        super().__init__()
        self.x_dim = x_dim
        self.recon_loss_type = recon_loss_type.lower()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims

        if activation is None:
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = activation()

        # ---------- Encoder ----------
        modules = []
        in_channels = x_dim
        for h_dim in hidden_dims:
            modules.append(nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1))
            modules.append(nn.BatchNorm2d(h_dim))
            modules.append(self.act)
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        # Determine flatten size after encoder
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            enc_out = self.encoder(dummy)
            self.enc_out_shape = enc_out.shape[1:]  # (C, H, W)
            self.enc_flat_dim = enc_out.numel() // enc_out.shape[0]

        self.fc_mu = nn.Linear(self.enc_flat_dim, latent_dim)
        self.fc_var = nn.Linear(self.enc_flat_dim, latent_dim)

        # ---------- Decoder ----------
        self.decoder_input = nn.Linear(latent_dim, self.enc_flat_dim)
        decoder_modules = []
        reversed_hidden_dims = list(reversed(hidden_dims))
        in_channels = self.enc_out_shape[0]
        for i in range(len(reversed_hidden_dims) - 1):
            decoder_modules.append(
                nn.ConvTranspose2d(
                    in_channels, reversed_hidden_dims[i + 1],
                    kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            decoder_modules.append(nn.BatchNorm2d(reversed_hidden_dims[i + 1]))
            decoder_modules.append(self.act)
            in_channels = reversed_hidden_dims[i + 1]

        self.decoder = nn.Sequential(*decoder_modules)
        # Output layer, project back to input channel count
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, x_dim, kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.Tanh()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes input and returns (mu, log_var)
        Args:
            x [B, C, H, W]: Input
        Returns:
            mu: [B, D]
            log_var: [B, D]
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent vector z to image space.
        Args:
            z [B, D]
        Returns:
            recon_x [B, C, H, W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, *self.enc_out_shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass
        Args:
            x [B, C, H, W]
        Returns:
            recon_x, x, mu, log_var
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, x, mu, log_var

    def loss_function(
        self, 
        recons: torch.Tensor, 
        input: torch.Tensor,
        mu: torch.Tensor, 
        log_var: torch.Tensor, 
        kld_weight: float = 1.0, 
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
        if self.recon_loss_type == 'bce':
            recon_loss = F.binary_cross_entropy(recons, input, reduction='mean')
        else:
            recon_loss = F.mse_loss(recons, input)

        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        loss = recon_loss + kld_weight * kld
        return VAEOutput(loss=loss, recon_loss=recon_loss.detach(), kld=kld.detach())

    def sample(self, 
               batch_size: int, 
               device: torch.device, 
               **kwargs: Any
        ) -> torch.Tensor:
        """
        Generate random samples from the VAE's latent space.
        """
        z = torch.randn(batch_size, self.fc_mu.out_features).to(device)
        samples = self.decode(z)
        return samples

    def generate(self, 
                 x: torch.Tensor, 
                 **kwargs: Any
        ) -> torch.Tensor:
        """
        For reconstruction of inputs.
        """
        return self.forward(x)[0]