import torch 
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

class VAE(BaseVAE):
    def __init__(self
                 , x_dim: int
                 , hidden_dim: List[int]
                 , latent_dim: int 
                 , **kwargs) -> None:
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        '''
        Encoder: Maps the input to a Gaussian distribution in latent space.
        The encoder consists of several fully connected layers with Leaky ReLU activation functions.
        '''
        modules = []
        if hidden_dim is None:
            hidden_dim = [32, 64, 128, 256, 512]
        
        in_channels = x_dim
        for h_dim in hidden_dim:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels = h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dim[-1] * 4, latent_dim)  # Mean of the Gaussian distribution
        self.fc_var = nn.Linear(hidden_dim[-1] * 4, latent_dim)

        '''
        Decoder: Maps the latent representation back to the original data space.
        The decoder consists of several fully connected layers with Leaky ReLU activation functions.
        '''
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dim[-1] * 4)
        hidden_dim.reverse()
        for i in  range(len(hidden_dim) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dim[i], hidden_dim[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dim[i + 1]),
                    nn.LeakyReLU(0.2)
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[-1], hidden_dim[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim[-1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim[-1], out_channels=x_dim, kernel_size=3, stride=1, padding=1),
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

    def decode(self, z: Tensor) -> Any:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
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