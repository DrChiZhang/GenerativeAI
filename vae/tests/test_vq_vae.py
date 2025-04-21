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

from models.vq_vae import VQVAE, VectorQuantizer

class TestVQVAE(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_channels = 3
        self.img_size = 32
        self.embedding_dim = 8
        self.num_embeddings = 16
        self.hidden_dims = [16, 32]
        self.vqvae = VQVAE(
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            num_embeddings=self.num_embeddings,
            hidden_dims=self.hidden_dims,
            img_size=self.img_size
        )

    def test_vector_quantizer_shapes_and_loss(self):
        vq = VectorQuantizer(self.num_embeddings, self.embedding_dim)
        z = torch.randn(self.batch_size, self.embedding_dim, 4, 4)
        quantized, vq_loss = vq(z)
        self.assertEqual(quantized.shape, z.shape)
        self.assertTrue(isinstance(vq_loss.item(), float))
        self.assertTrue(torch.isfinite(vq_loss))

    def test_end_to_end_forward(self):
        imgs = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size)
        output = self.vqvae(imgs)
        recons, input_, vq_loss = output
        self.assertEqual(recons.shape, imgs.shape)
        self.assertEqual(input_.shape, imgs.shape)
        self.assertTrue(isinstance(vq_loss.item(), float))

    def test_codebook_indices_shape(self):
        imgs = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size)
        indices = self.vqvae.get_codebook_indices(imgs)
        self.assertEqual(indices.shape[0], self.batch_size)
        self.assertTrue(indices.dtype in (torch.long, torch.int64, torch.int32))

    def test_loss_function_output(self):
        imgs = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size)
        output = self.vqvae(imgs)
        vae_output = self.vqvae.loss_function(*output)
        self.assertTrue(hasattr(vae_output, "loss"))
        self.assertTrue(hasattr(vae_output, "recon_loss"))
        self.assertTrue(hasattr(vae_output, "vq_loss"))
        self.assertTrue(torch.isfinite(vae_output.loss))
        self.assertTrue(torch.isfinite(vae_output.recon_loss))
        self.assertTrue(torch.isfinite(vae_output.vq_loss))

    def test_summary(self):
        """
        Test that torchsummary.summary runs on the VQ-VAE model without errors.
        """
        try:
            summary(
                self.vqvae,
                input_size=(self.in_channels, self.img_size, self.img_size),
                device="cpu"
            )
        except Exception as e:
            self.fail(f"torchsummary.summary failed: {e}")

if __name__ == '__main__':
    unittest.main()