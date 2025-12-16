import logging
from typing import Optional


def get_logger(name: Optional[str] = None, level = logging.INFO) -> logging.Logger:
    log_format = (
        "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s"
    )
    
    date_format = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger(name or __name__)
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger


log = get_logger("GlobalInfoLogger")
