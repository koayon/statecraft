import getpass
import os

import torch as t


def get_default_cache_dir():
    if os.name == "posix":  # For Linux and macOS
        return os.path.expanduser("~/.cache/statecraft/")
    elif os.name == "nt":  # For Windows
        try:
            username = getpass.getuser()
            return os.path.expanduser(rf"C:\Users\{username}\.cache\statecraft")
        except:  # Test windows machine has no username so return current directory
            return os.path.expanduser(rf".cache\statecraft")
    else:
        raise ValueError(f"Unsupported operating system: {os.name}")


def default_device():
    if t.cuda.is_available():
        device = "cuda"
    if t.backends.mps.is_available():
        # Running on Apple Silicon (M1, M2, etc. chip)
        device = "mps"
    else:
        device = "cpu"
    return t.device(device)
