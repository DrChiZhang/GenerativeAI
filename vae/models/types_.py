import torch
from typing import List, Callable, Union, Any, TypeVar, Tuple

from dataclasses import dataclass

Tensor = TypeVar('torch.tensor')

@dataclass
class VAEOutput:
    loss: torch.Tensor = None
    recon_loss: torch.Tensor = None
    kld_loss: torch.Tensor = None
    vq_loss: torch.Tensor = None