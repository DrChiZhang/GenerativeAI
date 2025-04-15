import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  # Go up 2 levels from tests/
sys.path.append(str(project_root))

import torch 
import unittest
from models.vae import VAE
from torchsummary import summary

class TestVAE(unittest.TestCase):
    def setUp(self):
        # Set up a VAE model for testing
        self.input_dim = 3  # Example input dimension (e.g., for MNIST)
        self.hidden_dim = [32, 64, 128, 256, 512]
        self.latent_dim = 10
        self.model = VAE(self.input_dim, self.hidden_dim, self.latent_dim)

    def test_vae_initialization(self):
        # Test if the model initializes correctly
        self.assertIsInstance(self.model.encoder, torch.nn.Module)
        self.assertIsInstance(self.model.decoder, torch.nn.Module)

    def test_vae_forward(self):
        # Test the forward pass of the VAE
        x = torch.randn(16, 3, 64, 64)
        y = self.model(x)
        self.assertEqual(y[0].size(), (16, 3, 64, 64))  # Check the output size
        self.assertEqual(y[1].size(), (16, 3, 64, 64))
        self.assertEqual(y[2].size(), (16, self.latent_dim))
        self.assertEqual(y[3].size(), (16, self.latent_dim))

    def test_vae_loss(self):
        # Test the loss function
        x = torch.randn(16, 3, 64, 64)

        result = self.model(x)
        loss = self.model.loss_function(result[0], result[1], result[2], result[3], kld_weight = 0.005)
        print(loss)

    def test_vae_summary(self):
        # Test the model summary
        print(summary(self.model, (3, 64, 64), device='cpu'))
        # Check if the summary prints without errors
        # Note: The summary function prints the model architecture and parameters
        # We can't assert the printed output, but we can check if it runs without error

if __name__ == '__main__':
    unittest.main()