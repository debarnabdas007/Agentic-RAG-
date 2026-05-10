import logging
import sys
import os
from backend.config import settings

def setup_logger(name: str) -> logging.Logger:

    """  Going to write on both the console and the file  """
    
    # To Ensure log directory exists
    os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if the logger is requested multiple times !!
    if logger.handlers:
        return logger

    # Formatting
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ## Console Handler (Prints to terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    ## File Handler (Writes to logs/agent.log)
    file_handler = logging.FileHandler(settings.LOG_FILE_PATH, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger




## NOTE: 
'''
if logger.handlers:
    return logger

This prevents duplicate handlers !!
Without this:
> importing logger multiple times
> duplicates every log line
Classic Python logging bug that I had faced !!
'''