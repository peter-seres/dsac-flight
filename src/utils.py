""" utility functions for pytorch, RNG state, argparse"""

import random
import numpy as np
import torch


def get_device(use_cuda: bool = False) -> torch.device:
    """Fetch the torch device. Check if cuda is wanted or available."""
    return torch.device("cuda:0" if (torch.cuda.is_available() and use_cuda) else "cpu")


def get_cpu() -> torch.device:
    """Fetch the torch cpu device."""
    return torch.device("cpu")


def get_cuda(fallback_to_cpu: bool = False) -> torch.device:
    """Fetch the CUDA device. If the fallback is true, returns the CPU, otherwise raises an error."""
    if not torch.cuda.is_available():
        if fallback_to_cpu:
            return get_cpu()
        else:
            raise ValueError("Requested but couldn't find device.")
    return torch.device("cuda:0")


def control_randomness(seed: int) -> None:
    """Set global randomness seeds for python, numpy and pytorch."""

    # Python
    random.seed(seed)

    # Pytorch
    torch.manual_seed(seed)

    # Numpy
    np.random.seed(seed)


def str2bool(val: str) -> bool:
    """
    Convert a string representation of truth to true (1) or false (0).
    - True values: 'y', 'yes', 't', 'true', 'on', '1'
    - False values: 'n', 'no', 'f', 'false', 'off', '0'
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"Cannot parse invalid bool string value: {val}")
