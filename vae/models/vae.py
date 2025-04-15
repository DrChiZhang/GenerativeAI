import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

class VAE(BaseVAE):
    def __init__(self, x_dim: int, hidden_dim: List[int], latent_dim: int, **kwargs) -> None:
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Default hidden dimensions
        if hidden_dim is None:
            hidden_dim = [32, 64, 128, 256, 512]

        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = self._build_encoder(x_dim)

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dim[-1] * 4, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim[-1] * 4)
        self.decoder = self._build_decoder()
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[0], hidden_dim[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim[0], x_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def _build_encoder(self, in_channels: int) -> nn.Sequential:
        modules = []
        for h_dim in self.hidden_dim:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
        return nn.Sequential(*modules)

    def _build_decoder(self) -> nn.Sequential:
        modules = []
        reversed_hidden_dim = list(reversed(self.hidden_dim))
        for i in range(len(reversed_hidden_dim) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(reversed_hidden_dim[i], reversed_hidden_dim[i + 1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(reversed_hidden_dim[i + 1]),
                    nn.LeakyReLU(0.2)
                )
            )
        return nn.Sequential(*modules)

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Encodes input and returns latent codes.
        :param x: Input tensor [N x C x H x W]
        :return: List of mean and log variance tensors
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
    
    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dim[-1], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var).
        :param mu: Mean tensor
        :param log_var: Log variance tensor
        :return: Sampled tensor
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        """
        Forward pass through the network.
        :param x: Input tensor [B x C x H x W]
        :return: List containing reconstructed image, input image, mean, and log var
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def loss_function(
        self,
        recons: Tensor,
        input: Tensor,
        mu: Tensor,
        log_var: Tensor,
        kld_weight: float
    ) -> dict:
        """
        Computes the VAE loss function.
        :param recons: Reconstructed tensor
        :param input: Original input tensor
        :param mu: Latent mean
        :param log_var: Latent log variance
        :param kld_weight: KLD weight
        :return: Loss components
        """
        recons_loss = F.mse_loss(recons, input)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        total_loss = recons_loss + kld_weight * kld_loss
        return {
            'loss': total_loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kld_loss.detach()
        }

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from latent space and reconstructs input space.
        :param batch_size: Number of samples
        :param current_device: Device for computation
        :return: Reconstructed tensor samples
        """
        z = torch.randn(batch_size, self.latent_dim).to(current_device)
        return self.decode(z)
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Generates a reconstruction of input image.
        :param x: Input image [B x C x H x W]
        :return: Reconstructed image [B x C x H x W]
        """
        return self.forward(x)[0]