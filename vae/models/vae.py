import torch 
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

class VAE(BaseVAE):
    def __init__(self, x_dim: int, hidden_dim: List[int], latent_dim: int, **kwargs) -> None:
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder setup remains the same
        modules = []
        if hidden_dim is None:
            hidden_dim = [32, 64, 128, 256, 512]
        self.hidden_dim = hidden_dim.copy()  # Store original hidden_dim
        
        in_channels = x_dim
        for h_dim in self.hidden_dim:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        
        # Save the encoder's last channel count
        self.encoder_last_channel = self.hidden_dim[-1]
        self.fc_mu = nn.Linear(self.encoder_last_channel * 4, latent_dim)
        self.fc_var = nn.Linear(self.encoder_last_channel * 4, latent_dim)
        
        # Decoder setup
        self.decoder_input = nn.Linear(latent_dim, self.encoder_last_channel * 4)
        
        # Reverse hidden_dim for decoder
        reversed_hidden_dim = list(reversed(self.hidden_dim))
        modules = []
        for i in range(len(reversed_hidden_dim) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(reversed_hidden_dim[i], reversed_hidden_dim[i+1], 
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(reversed_hidden_dim[i+1]),
                    nn.LeakyReLU(0.2)
                )
            )
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(reversed_hidden_dim[-1], reversed_hidden_dim[-1], 
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(reversed_hidden_dim[-1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(reversed_hidden_dim[-1], x_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        logvar = self.fc_var(result)

        return [mu, logvar]
    
    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        # Use encoder's last channel count for reshaping
        result = result.view(-1, self.encoder_last_channel, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return [self.decode(z), x, mu, logvar]

    
    def loss_function(
        self,
        recons: torch.Tensor,
        input: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        kld_weight: float  # Pass M_N as kld_weight
    ) -> dict:
        """
        Computes the VAE loss function:
        KL(N(μ, σ) || N(0, 1)) = 0.5 * Σ(1 + log(σ²) - μ² - σ²)
        
        Args:
            recons: Reconstructed data (decoder output)
            input: Original input data
            mu: Latent mean from encoder
            log_var: Latent log-variance from encoder
            kld_weight: Weight for KLD term (M_N in original code)

        Returns:
            dict: Loss components for logging.
        """
        # Reconstruction loss (MSE)
        recons_loss = F.mse_loss(recons, input)

        # KL Divergence (mean over batch)
        kld_loss = 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kld_loss = torch.mean(kld_loss)  # Mean over batch

        # Total loss
        total_loss = recons_loss + kld_weight * kld_loss

        return {
            'loss': total_loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kld_loss.detach()  # No negative sign!
        }


    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param batch_size: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(batch_size, self.latent_dim).to(current_device)
        return self.decode(z)
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]