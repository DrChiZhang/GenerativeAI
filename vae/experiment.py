import os
import torch
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Any, Dict, Optional
import torchvision.utils as vutils

class VAEXperiment(pl.LightningModule):
    """
    PyTorch LightningModule for Variational Autoencoder (VAE) experiments.
    Compatible with a VAE whose loss_function returns a VAEOutput dataclass
    containing loss, recon_loss, and kld terms.
    """

    def __init__(
        self,
        vae_model: torch.nn.Module,
        params: Dict[str, Any]
    ) -> None:
        """
        Args:
            vae_model: A VAE model that implements __call__, loss_function, and sample.
            params: Dictionary of training hyperparameters and experiment settings.
        """
        super().__init__()
        self.model = vae_model          # Save the VAE model
        self.params = params            # Store hyperparameters/settings dictionary
        self.curr_device: Optional[torch.device] = None
        self.hold_graph = self.params.get('retain_first_backpass', False)  # For advanced use cases

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        """
        Defines the forward pass for Lightning (simply calls the model).
        """
        return self.model(input, **kwargs)  # Returns (recons, input, mu, log_var)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        """
        Performs a training step using a batch.
        - Runs forward pass
        - Calculates VAE loss (reconstruction + KL divergence)
        - Logs each component to the logger / progress bar

        Args:
            batch: (input_images, labels)
            batch_idx: Index of the current training batch

        Returns:
            Scalar total loss for optimizer.step().
        """
        images, labels = batch                   # Unpack batch
        self.curr_device = images.device         # Track device for later use (e.g., sampling)

        results = self.forward(images, labels=labels, optimizer_idx = 0)          # Forward pass returns (recons, input, mu, log_var)
        # Compute the loss. Use annealed KLD weight during training if provided.
        vae_loss = self.model.loss_function(
            *results,  # Unpack results: (recons, input_img, mu, log_var)
            kld_weight = self.params['kld_weight'],
            optimizer_idx = 0,
            batch_idx = batch_idx
        )
        # vae_loss is a VAEOutput dataclass with attributes: loss, recon_loss, kld

        # Log loss and metrics to TensorBoard and progress bar
        self.log("train_loss", vae_loss.loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_recon_loss", vae_loss.recon_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if vae_loss.kld_loss is not None:
            self.log("train_kld_loss", vae_loss.kld_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if vae_loss.vq_loss is not None:
            self.log("train_vq_loss", vae_loss.vq_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return vae_loss.loss                    # Return total loss for gradient calculation

    def validation_step(self, batch, batch_idx: int, optimizer_idx = 0) -> None:
        """
        Performs a validation step.
        - Forward pass
        - Computes and logs VAE losses
        - Uses fixed kld_weight (no annealing typically)

        Args:
            batch: (input_images, labels)
            batch_idx: Index of the current validation batch
        """
        images, labels = batch
        self.curr_device = images.device

        results = self.forward(images, labels=labels)          # Forward pass
        vae_loss = self.model.loss_function(
            *results,  # Unpack results: (recons, input_img, mu, log_var)
            kld_weight = self.params['kld_weight'],
            optimizer_idx = optimizer_idx,
            batch_idx = batch_idx
        )

        # Log loss and metrics (only by epoch, not per batch)
        self.log("val_loss", vae_loss.loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_recon_loss", vae_loss.recon_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if vae_loss.kld_loss is not None:
            self.log("train_kld_loss", vae_loss.kld_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if vae_loss.vq_loss is not None:
            self.log("train_vq_loss", vae_loss.vq_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of each validation epoch by Lightning.
        Triggers sampling/reconstruction image saving for visualization.
        """
        self.sample_images()
    """
    Pytorch Lightning's on_validation_epoch_end() method is called at the end of each validation epoch.
    It is used to perform any actions that should occur after the validation loop has completed.
    """
    def sample_images(self) -> None:
        """
        Samples from the test set to produce and save example reconstructions and
        (if implemented) generated samples from the prior.
        """
        dataloader = self.trainer.datamodule.test_dataloader()   # Get test dataloader
        test_input, test_label = next(iter(dataloader))          # Grab a batch
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        # Reconstruction: pass input through model
        recons, _, _, _ = self.model(test_input, labels=test_label)
        recons_dir = os.path.join(self.logger.log_dir, "Reconstructions")
        os.makedirs(recons_dir, exist_ok=True)
        vutils.save_image(
            recons.data,
            os.path.join(
                recons_dir,
                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"
            ),
            normalize=True,
            nrow=12                   # Number of images per row in the grid
        )

        # Generate random samples only if model supports .sample()
        """ 
        Save exactly num_samples generated images as a grid.
        The model.sample() method is expected to take num_classes, device, and labels as arguments.
        The labels are used to condition the generation process.
        If the test set has fewer than num_samples, repeat the test set labels as needed.
        """
        num_samples = self.params.get('num_samples', 144)  # Default to 144 samples
        if test_label.shape[0] < num_samples:
            # Repeat as needed and trim to right size
            reps = (num_samples + test_label.shape[0] - 1) // test_label.shape[0]
            sample_labels = test_label.repeat((reps, 1))[:num_samples]
        else:
            sample_labels = test_label[:num_samples]

        samples_dir = os.path.join(self.logger.log_dir, "Samples")
        os.makedirs(samples_dir, exist_ok=True)
        try:
            samples = self.model.sample(num_samples, self.curr_device, labels=sample_labels)
            vutils.save_image(
                samples.cpu().data,
                os.path.join(
                    samples_dir,
                    f"samples_{self.logger.name}_Epoch_{self.current_epoch}.png"
                ),
                normalize=True,
                nrow=12
            )
        except (NotImplementedError, AttributeError):
            # In case sampling from prior is not implemented
            self.print("WARNING: The model does not implement a .sample method.")

    def configure_optimizers(self):
        """
        Sets up optimizer(s) and learning rate scheduler(s) for Lightning.
        Handles main optimizer and possible secondary optimizer (for GAN/Adversarial losses).

        Returns:
            List of optimizers, optionally paired with LR schedulers.
        """
        optimizers = []
        schedulers = []

        # Main optimizer for all VAE parameters
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params.get('LR', 1e-3),
            weight_decay=self.params.get('weight_decay', 0)
        )
        optimizers.append(optimizer)

        # (Optional) Second optimizer, if specified (e.g., for a submodel like a discriminator)
        lr2 = self.params.get('LR_2')
        submodel_name = self.params.get('submodel')
        if lr2 and submodel_name and hasattr(self.model, submodel_name):
            submodel = getattr(self.model, submodel_name)
            optimizer2 = optim.Adam(submodel.parameters(), lr=lr2)
            optimizers.append(optimizer2)

        # Main LR scheduler (optional)
        gamma = self.params.get('scheduler_gamma')
        if gamma is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            schedulers.append(scheduler)

        # (Optional) Scheduler for the second optimizer
        gamma2 = self.params.get('scheduler_gamma_2')
        if len(optimizers) > 1 and gamma2 is not None:
            scheduler2 = optim.lr_scheduler.ExponentialLR(optimizers[1], gamma=gamma2)
            schedulers.append(scheduler2)

        # Lightning expects (optimizers, schedulers) if any schedulers exist, otherwise just optimizers
        return (optimizers, schedulers) if schedulers else optimizers