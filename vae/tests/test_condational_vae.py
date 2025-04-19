import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import unittest

try:
    from torchsummary import summary
    has_summary = True
except ImportError:
    has_summary = False

from models.conditional_vae import ConditionalVAE, VAEOutput


class TestConditionalVAE(unittest.TestCase):
    def setUp(self):
        # Define small testable config
        self.batch_size = 4
        self.num_classes = 10
        self.latent_dim = 8
        self.img_size = 32
        self.in_channels = 3
        # Instantiate model
        self.model = ConditionalVAE(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            latent_dim=self.latent_dim,
            hidden_dims=[16, 32],
            img_size=self.img_size
        )

    def test_forward_and_loss(self):
        # Create random images and one-hot labels
        x = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size)
        labels = torch.zeros(self.batch_size, self.num_classes)
        label_indices = torch.randint(0, self.num_classes, (self.batch_size,))
        labels[torch.arange(self.batch_size), label_indices] = 1

        # Forward pass
        recons, inp, mu, log_var = self.model(x, labels)

        # Check output shapes
        self.assertEqual(recons.shape, x.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(log_var.shape, (self.batch_size, self.latent_dim))

        # Compute VAEOutput
        vae_output = self.model.loss_function(recons, inp, mu, log_var, M_N=1.0)
        self.assertIsInstance(vae_output, VAEOutput)
        # Loss should be a scalar
        self.assertEqual(vae_output.loss.shape, torch.Size([]))
        self.assertIsInstance(vae_output.loss.item(), float)

        # Check that losses are finite
        self.assertTrue(torch.isfinite(vae_output.loss))
        self.assertTrue(torch.isfinite(vae_output.recon_loss))
        self.assertTrue(torch.isfinite(vae_output.kld_loss))

    def test_model_summary(self):
        # Print a summary of the encoder module
        # (first arg: model, second arg: input size, including channels)
        try:
            summary(
                self.model,
                input_size=[(self.in_channels, self.img_size, self.img_size),
                            (self.num_classes,)],  # because forward expects both x and label
                dtypes=[torch.float, torch.float],  # datatypes for x and label
                batch_size=self.batch_size,
                device="cpu"
            )
        except Exception as e:
            self.fail(f"Model summary failed with error: {e}")

if __name__ == '__main__':
    unittest.main()