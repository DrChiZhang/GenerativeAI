from .base import * 
from .vae import * 
from .conditional_vae import *
from .vq_vae import *

"""
VAE Models Dictionary
Define a dictionary that maps model names (strings) to their corresponding class constructors.
This allows dynamic model selection and instantiation by name.
Usage Example:
    model_name = "VAE"
    model_class = vae_models[model_name]
    model_instance = model_class(**kwargs)
This is useful for loading different models based on user input or configuration files.
"""
vae_models = {
    "VAE": VAE,
    "ConditionalVAE": ConditionalVAE,
    "VQVAE": VQVAE,
}

from .pixel_cnn import PixelCNN

"""
PixelCNN Models Dictionary
Define a dictionary that maps model names (strings) to their corresponding class constructors.
This allows dynamic model selection and instantiation by name.
Usage Example:
    model_name = "PixelCNN"
    model_class = pixelcnn_models[model_name]
    model_instance = model_class(**kwargs)
"""
ar_models = {
    "PixelCNN": PixelCNN,   
}