import os
import yaml

class Config:

    def __init__(self,config_path='../../../configs/config-alltogether.yaml'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path,'r') as f:
            self.config = yaml.safe_load(f)

    def model_config(self):
        return self.config['model']

    def training_config(self):
        return self.config['training']

    def optuna_config(self):
        return self.config['optuna']
    def to_dict(self):
        return self.config
    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        config.config = config_dict
        return config
        