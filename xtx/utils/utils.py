import random
import torch
import numpy as np
from datetime import datetime
import os

# Note that the Z refers to the UTC (Coordinated Universal Time)
# The advantage of UTC is that it does not have a timezone and is thus unambiguous
DATE_TIME_FORMAT: str = '%Y-%m-%dT%H:%M:%SZ'


def set_random_seed(seed: int):
    """ Set all random seeds for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_safe_output_dir(path):
    """ Return a unique local path if none is provided """
    if path is not None:
        return path
    else:
        return os.path.join('./logs/', datetime.now().strftime(DATE_TIME_FORMAT))


def create_directory(path):
    """ Creates directory if it doesn't exist """
    os.makedirs(path, exist_ok=True)
    return path
