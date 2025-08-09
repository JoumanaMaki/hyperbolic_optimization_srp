import random
import os

import numpy as np
import torch

from config import SEED


def seed_everything() -> None:
    """
    Makes experiments reproducible.
    To work in jupyter notebook it must be run in every cell that contains randomized functions.
    """
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
