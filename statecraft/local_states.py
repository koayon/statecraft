import os
from typing import Optional

from statecraft import StatefulModel

ALL_DIRS_TO_REMOVE = [".DS_Store"]


def list_local_states(model_name: str, cache_dir: Optional[str] = None) -> list[str]:
    if cache_dir is None:
        cache_dir = StatefulModel._get_default_cache_dir()

    out = []

    model_dir = os.path.join(cache_dir, model_name, "CURRENT_USER")
    if os.path.isdir(model_dir):
        current_user_states = os.listdir(model_dir)
        out += [dir for dir in current_user_states if dir not in ALL_DIRS_TO_REMOVE]

    other_dirs = os.listdir(os.path.join(cache_dir, model_name))
    other_dirs = [
        dir
        for dir in other_dirs
        if (dir not in ALL_DIRS_TO_REMOVE) and (dir != "CURRENT_USER")
    ]

    for dir in other_dirs:
        if dir not in ("CURRENT_USER", ".DS_Store"):
            sub_dirs = os.listdir(os.path.join(cache_dir, model_name, dir))
            out += [
                f"{dir}/{sub_dir}" for sub_dir in sub_dirs if sub_dir not in ALL_DIRS_TO_REMOVE
            ]

    return out


if __name__ == "__main__":
    states = list_local_states("state-spaces/mamba-130m-hf")
    print(states)
