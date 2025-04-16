from pathlib import Path
from typing import List, Optional, Sequence, Union, Any
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA
import pytorch_lightning as pl

class MyCelebA(CelebA):
    """
    Workaround to always pass integrity check for CelebA Dataset.
    Override _check_integrity to always return True.
    """
    def _check_integrity(self) -> bool:
        return True

class VAEDataset(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for VAE training on CelebA dataset.
    
    Args:
        data_dir (str): Directory where the dataset is stored.
        train_batch_size (int): Batch size for training loader.
        val_batch_size (int): Batch size for validation loader.
        patch_size (int or tuple): Resize images to this size (H, W).
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory during data loading.
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.data_dir = str(data_dir)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Define transform composition
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up train and validation datasets. 
        """
        self.train_dataset = MyCelebA(
            self.data_dir, split='train', transform=self.transforms, download=False
        )
        self.val_dataset = MyCelebA(
            self.data_dir, split='test', transform=self.transforms, download=False
        )

    def train_dataloader(self) -> DataLoader:
        """Returns dataloader for training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Returns dataloader for validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Returns dataloader for test set. Batch size is same as val_batch_size by default."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )