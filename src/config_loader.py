"""
src/config_loader.py
====================
Loads and validates the YAML configuration file.
Single source of truth — all modules pull config from here.
"""
 
import yaml
from pathlib import Path
from src.logger import get_logger
 
logger = get_logger(__name__)
 
 
def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load YAML config file and return as nested dict.
 
    Parameters
    ----------
    config_path : str
        Path to the YAML config file.
 
    Returns
    -------
    dict
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f'Config loaded from: {config_path}')
    logger.info(f"Project: {config['project']['name']} v{config['project']['version']}")
    
    return config