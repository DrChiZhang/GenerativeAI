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
    parser.add_argument('--config', '-c', dest="filename", metavar='FILE',
                        help='Path to the config file', default='configs/vae.yaml')
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
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),  # Record learning rate over epochs
        ModelCheckpoint(
            save_top_k=2,
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            filename='{epoch}-{val_loss:.4f}',         # Name includes validation loss
            monitor="val_loss",
            mode="min",                                # Lower val_loss is better
            save_last=True,
        ),
    ]

    # --------------------- Initialize and run the trainer ---------------------
    runner = Trainer(
        logger=tb_logger,    # Attach TensorBoard logger
        callbacks=callbacks, # Attach hooks for LR and checkpointing
        accelerator='gpu', 
        devices=1
    )

    print(f"======= Training {model_name} =======")
    runner.fit(experiment, datamodule=data)   # Start training

if __name__ == "__main__":
    main()