import copy
import getpass
import os

import torch as t
from transformers.models.mamba.modeling_mamba import MambaCache


def get_default_cache_dir():
    if os.name == "posix":  # For Linux and macOS
        return os.path.expanduser("~/.cache/statecraft/")
    elif os.name == "nt":  # For Windows
        username = getpass.getuser()
        return os.path.expanduser(rf"C:\Users\{username}\.cache\statecraft")
    else:
        raise ValueError(f"Unsupported operating system: {os.name}")


def cache_to_device(cache: MambaCache, device: str) -> MambaCache:
    mamba_cache = copy.deepcopy(cache)

    ssm_states = {
        layer_num: layer_cache.to(device)
        for layer_num, layer_cache in mamba_cache.ssm_states.items()
    }

    conv_states = {
        layer_num: layer_cache.to(device)
        for layer_num, layer_cache in mamba_cache.conv_states.items()
    }

    mamba_cache.ssm_states = ssm_states
    mamba_cache.conv_states = conv_states

    return mamba_cache
