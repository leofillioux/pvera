import os
import yaml
from easydict import EasyDict as edict

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = edict(yaml.safe_load(f))
    return config

adapters = read_config(os.path.join(os.path.dirname(__file__), 'adapters.yaml'))
dataset = read_config(os.path.join(os.path.dirname(__file__), 'datasets.yaml'))
training = read_config(os.path.join(os.path.dirname(__file__), 'training.yaml'))