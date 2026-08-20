import yaml
from src.logger import get_logger

logger = get_logger(__name__)

def load_config(config_path: str = "config/config.yaml") -> dict:

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logger.info(f"Configuration loaded from {config_path}")
    return config