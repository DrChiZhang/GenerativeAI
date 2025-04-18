import os
import yaml
import argparse
from pathlib import Path

from models import *                  # vae_models dict should be defined here
from experiment import VAEXperiment   # Your experiment module
from dataset import VAEDataset        # Your datamodule implementation

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything  
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

def load_config(config_path):
    """
    Safe YAML config loading from given path.

    Args:
        config_path (str): path to configuration YAML file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Error parsing YAML config:", exc)
            exit(1)  # Exit if YAML is malformed

def main():
    # --------------------- Parse arguments ---------------------
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c', dest="filename", metavar='FILE', help='Path to the config file', default='configs/vae.yaml')
    args = parser.parse_args()

    # --------------------- Load configuration ---------------------
    config = load_config(args.filename)

    # --------------------- Set up logging ---------------------
    # Will save TensorBoard logs under save_dir/model_name/
    tb_logger = TensorBoardLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['model_params']['name'],
    )

    # --------------------- Set seed for reproducibility ---------------------
    manual_seed = config['exp_params'].get('manual_seed', 42)
    seed_everything(manual_seed, True)

    # --------------------- Build model ---------------------
    # Ensure the requested model exists
    model_name = config['model_params']['name']
    if model_name not in vae_models:
        raise ValueError(f"Model '{model_name}' not found in vae_models dict!")
    # Instantiate model using model_params from config
    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model, config['exp_params'])
    """
    Python knowledge:
    The ** operator in Python is unpacking a dictionary into keyword arguments when calling a function or instantiating a class.
    Usage:
        params = {'a': 1, 'b': 2}
        func(**params)   # Equivalent to func(a=1, b=2)
    Note:  the ** operator only works with mapping types (mainly dict, or custom mapping objects).
    """

    # --------------------- Data module setup ---------------------
    # Pin memory if using GPU workers
    num_gpus = config['trainer_params'].get('gpus', 0)
    pin_memory = bool(num_gpus)
    data = VAEDataset(**config["data_params"], pin_memory=pin_memory)
    data.setup()  # Prepare and split data

    # --------------------- Prepare output directories ---------------------
    # Make sure directories for samples, reconstructions, and checkpoints exist
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/checkpoints").mkdir(exist_ok=True, parents=True)

    # --------------------- Prepare PyTorch Lightning Callbacks ---------------------
    """
    Create a list of callback objects that will be passed to a training framework (likely PyTorch Lightning).
    Callbacks are used to perform specific actions during model training, like saving checkpoints or monitoring metrics.
    """
    callbacks = [
        # Callback that logs the learning rate at the end of every epoch.
        # Useful for tracking dynamic learning rate schedules and debugging training issues.
        LearningRateMonitor(logging_interval='epoch'),  
        
        # Callback that saves model checkpoints during training.
        ModelCheckpoint(
            save_top_k=2,  # Keep only the best 2 checkpoints with the lowest validation loss.
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),  # Directory to save checkpoint files, organized under TensorBoard logger's run directory.
            filename='{epoch}-{val_loss:.4f}',  # Use epoch number and rounded validation loss in the checkpoint file name for easy identification.
            monitor="val_loss",         # Which metric to use for identifying the "best" model(s).
            mode="min",                 # "min" means that lower val_loss is better; use "max" for metrics where higher is better.
            save_last=True,             # Also save a checkpoint of the very last epoch, regardless of performance.
        ),
    ]

    # --------------------- Initialize and run the trainer ---------------------
    """
     PyTorch Lightning's Trainer is a high-level interface for organizing and running deep learning training loops. 
     The Trainer class controls how your model is trained, validated, and tested (not the model itself).
    """
    runner = Trainer(
        logger=tb_logger,    # Attach TensorBoard logger
        callbacks=callbacks, # Attach hooks for LR and checkpointing
        **config['trainer_params']  # Unpack trainer parameters from config
    )

    print(f"======= Training {model_name} =======")
    runner.fit(experiment, datamodule=data)   # Start training

if __name__ == "__main__":
    main()