from .types import * 
from torch import nn
from abc import abstractmethod 

class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()
    
    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Encodes the input data into a latent representation.
        
        Args:
            x (Tensor): Input data tensor.
        
        Returns:
            Tuple[Tensor]: Mean and log variance of the latent distribution.
        """
        raise NotImplementedError("Encoder method not implemented.")
    
    def decode(self, z: Tensor) -> Any:
        """
        Decodes the latent representation back to the original data space.
        
        Args:
            z (Tensor): Latent representation tensor.
        
        Returns:
            Any: Reconstructed data tensor.
        """
        raise NotImplementedError("Decoder method not implemented.")
    
    def sample(self, batch_szie: int, current_device:int, **kwargs) -> Tensor:
        """
        Samples from the latent space.
        
        Args:
            batch_size (int): Number of samples to generate.
            current_device (int): Current device index.
        
        Returns:
            Tensor: Sampled latent representation tensor.
        """
        raise NotImplementedError("Sampling method not implemented.")
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Generates data from the model.
        
        Args:
            x (Tensor): Input data tensor.
        
        Returns:
            Tensor: Generated data tensor.
        """
        raise NotImplementedError("Generate method not implemented.")
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (Tensor): Input data tensor.
        
        Returns:
            Tensor: Mean and log variance of the latent distribution.
        """
        pass
    
    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        """
        Computes the loss function for the model.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            Tensor: Loss and additional metrics.
        """
        pass