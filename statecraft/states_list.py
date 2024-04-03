import os
from typing import Optional

from statecraft import StatefulModel
from statecraft.client import client

ALL_DIRS_TO_REMOVE = [".DS_Store"]


def _list_local_states(model_name: str, cache_dir: Optional[str] = None) -> list[str]:
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
        if (dir not in ALL_DIRS_TO_REMOVE + ["CURRENT_USER"]) and (os.path.isdir(dir)):
            sub_dirs = os.listdir(os.path.join(cache_dir, model_name, dir))
            out += [
                f"{dir}/{sub_dir}" for sub_dir in sub_dirs if sub_dir not in ALL_DIRS_TO_REMOVE
            ]

    return out


def _list_server_states(model_name: str) -> list[str]:
    return client.get_states(model_name)


def show_available_states(model_name: str, cache_dir: Optional[str] = None) -> None:
    local_states = _list_local_states(model_name, cache_dir)
    server_states = _list_server_states(model_name)

    print(f"Local states for {model_name}:")
    print(local_states)

    print("")
    print(f"Server states for {model_name}:")
    print(server_states)


if __name__ == "__main__":
    states = _list_local_states("state-spaces/mamba-130m-hf")
    print(states)
