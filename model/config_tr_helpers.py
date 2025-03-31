import os
import yaml

class Config:
    def __init__(self, config_path=None):
        # If no config_path is provided, build one relative to this file's directory.
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, 'config_train.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def model_config(self):
        return self.config['model']

    def training_config(self):
        return self.config['training']

    def optuna_config(self):
        return self.config['optuna']

    # Add to_dict method
    def to_dict(self):
        return self.config

    # Add from_dict class method
    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        config.config = config_dict
        return config
