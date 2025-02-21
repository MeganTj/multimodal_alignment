import os
import sys
import random
import numpy as np
import torch
import json

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_args_to_file(args, filename):
    """
    Save argparse arguments to a file as JSON.
    
    :param args: Parsed arguments (Namespace object).
    :param filename: Path to the file where arguments should be saved.
    """
    with open(filename, 'w') as file:
        json.dump(vars(args), file, indent=4)