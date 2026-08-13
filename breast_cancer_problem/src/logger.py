"""
src/logger.py
=============
Centralised logging setup for the entire pipeline.
Import this in every module — never use print() in production code.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def get_logger(name: str, log_file: str = 'logs/pipeline.log', level: str = 'INFO') -> logging.Logger:

    """
    Factory function: returns a configured logger.
 
    Parameters
    ----------
    name : str
        Logger name — use __name__ in every module.
    log_file : str
        Path to the log file.
    level : str
        Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL
 
    Returns
    -------
    logging.Logger
    """

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger # avoid duplicate
    
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    #console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)

    #File handler
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger