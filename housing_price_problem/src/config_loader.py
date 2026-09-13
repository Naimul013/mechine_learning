import sys
from pathlib import Path
sys.path.append(str(Path.cwd().resolve()))
import yaml
from src.logger import get_logger

logger = get_logger(__name__)

def load_config(config_path: str | Path | None = None) -> dict:

    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / 'config' / 'config.yaml'
    else:
        config_path = Path(config_path).resolve()

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logger.debug(f"Configuration loaded from {config_path}")
    return config