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

from models.vae import VAE

class TestVAE(unittest.TestCase):
    def setUp(self):
        self.input_channels = 3
        self.input_size = 64
        self.latent_dim = 10
        # Setup for 64x64 test
        self.model = VAE(
            x_dim=self.input_channels,
            input_shape=(self.input_channels, self.input_size, self.input_size),
            hidden_dims=[32, 64, 128],
            latent_dim=self.latent_dim,
        )

    def test_vae_initialization(self):
        self.assertIsInstance(self.model.encoder, torch.nn.Module, "Encoder not instance of nn.Module")
        self.assertIsInstance(self.model.decoder, torch.nn.Module, "Decoder not instance of nn.Module")

    def test_vae_forward(self):
        batch = 8
        H = W = self.input_size
        x = torch.randn(batch, self.input_channels, H, W)
        recon_x, inp_x, mu, log_var = self.model(x)
        self.assertEqual(recon_x.shape, (batch, self.input_channels, H, W), "Reconstruction shape mismatch")
        self.assertEqual(inp_x.shape, (batch, self.input_channels, H, W), "Input passthrough shape mismatch")
        self.assertEqual(mu.shape, (batch, self.latent_dim), "Latent mean shape mismatch")
        self.assertEqual(log_var.shape, (batch, self.latent_dim), "Latent log_var shape mismatch")

    def test_vae_loss(self):
        batch = 4
        x = torch.randn(batch, self.input_channels, self.input_size, self.input_size)
        recon_x, inp_x, mu, log_var = self.model(x)
        loss_out = self.model.loss_function(recon_x, inp_x, mu, log_var, kld_weight=0.01)
        self.assertTrue(hasattr(loss_out, 'loss'), "Loss output missing 'loss' attribute")
        self.assertTrue(loss_out.loss.requires_grad, "Loss should be a differentiable tensor")
        self.assertGreaterEqual(loss_out.loss.item(), 0, "Loss should be non-negative")

    def test_vae_summary(self):
        if has_summary:
            try:
                summary(self.model, (self.input_channels, self.input_size, self.input_size), device='cpu')
            except Exception as e:
                self.fail(f"torchsummary.summary failed: {e}")
        else:
            print("torchsummary not installed; skipping summary test.")

    def test_enc_dec_shapes(self):
        """Test on another input shape to check dynamic adaptation."""
        model32 = VAE(
            x_dim=3, input_shape=(3, 32, 32),
            hidden_dims=[32, 64], latent_dim=5)
        x = torch.randn(2, 3, 32, 32)
        out, _, mu, log_var = model32(x)
        self.assertEqual(out.shape, (2, 3, 32, 32))
        self.assertEqual(mu.shape, (2, 5))
        self.assertEqual(log_var.shape, (2, 5))

if __name__ == '__main__':
    unittest.main()