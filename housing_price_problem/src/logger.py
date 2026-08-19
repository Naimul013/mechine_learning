import logging
import sys
from pathlib import Path



def get_logger(name: str, log_path: str = 'logs/pipeline.log', level: str = 'INFO')->logging.Logger:

    logger = logging.getLogger(name)
    path = Path(log_path).parent.mkdir(parents = True, exist_ok = True)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO)) # it will set the logging level to the specified level or default to INFO if the level is not recognized.

    if logger.handlers:
        return logger # avoid duplicate

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console handler
    console_handler = logging.streamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # file handler
    file_handler =logging.FileHandler(log_path, mode = 'a', encoding = 'utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

