import os
import sys

# Get the directory of the current script (train.py)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Determine the project root (assumes train.py is inside the 'model' folder)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
# Add the project root to sys.path if it isn't already
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
from typing import Dict, Optional
from utils import GrammarVAEModel

class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass

def load_config(config_path: str) -> Dict:
    """
    Load and validate configuration from YAML file.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate essential configurations
        required_keys = [
            'data.data_path',
            'training.batch_size',
            'training.epochs',
            'model.shared.z_dim',
            'training.monitor_metric'  # Make sure we know what to monitor
        ]
        
        for key in required_keys:
            sections = key.split('.')
            current = config
            for section in sections:
                if section not in current:
                    raise ConfigurationError(f"Missing required configuration: {key}")
                current = current[section]
                
        return config
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing config file: {e}")

def setup_callbacks(config: Dict) -> list:
    """
    Set up training callbacks based on configuration.
    """
    callbacks = []
    
    # Main checkpoint callback
    callbacks.append(
        ModelCheckpoint(
            monitor=config['training']['monitor_metric'],
            dirpath=config['saving']['base_dir'],
            filename=config['saving']['checkpoint_pattern'],
            save_top_k=3,
            mode='min',  # Assuming we're monitoring ELBO or loss
            save_last=True
        )
    )
    
    # Early stopping callback
    callbacks.append(
        EarlyStopping(
            monitor=config['training']['monitor_metric'],
            patience=config['training']['early_stopping_patience'],
            mode='min',
            verbose=True
        )
    )
    
    # Learning rate monitoring
    callbacks.append(
        LearningRateMonitor(logging_interval='step')
    )
    
    return callbacks

def train(config: Dict, resume_from_checkpoint: Optional[str] = None):
    """
    Main training function for the Grammar VAE.
    """
    # Create directories
    Path(config['saving']['base_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['saving']['logs_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = GrammarVAEModel(config)
    
    # Setup callbacks
    callbacks = setup_callbacks(config)
    
    # Setup logger
    logger = CSVLogger(
        save_dir=config['saving']['logs_dir'],
        name=config['saving']['model_name'],
        version=None,
        flush_logs_every_n_steps=10
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator=config['training']['accelerator'],
        devices=1,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config['training']['clip'],
        log_every_n_steps=10,
        precision=32,
        enable_progress_bar=True
    )
    
    # Train model
    trainer.fit(
        model,
        ckpt_path=resume_from_checkpoint
    )
    
    # Save final model
    final_path = os.path.join(
        config['saving']['base_dir'],
        f"{config['saving']['model_name']}_final.pth"
    )
    torch.save(model.state_dict(), final_path)
    
    return trainer, model

if __name__ == '__main__':
    # Build the configuration path relative to this file's directory
    config_path = os.path.join(script_dir, 'config_train.yaml')
    config = load_config(config_path)
    
    # Run training
    trainer, model = train(config)
