import getpass
import os


def get_default_cache_dir():
    if os.name == "posix":  # For Linux and macOS
        return os.path.expanduser("~/.cache/statecraft/")
    elif os.name == "nt":  # For Windows
        username = getpass.getuser()
        return os.path.expanduser(rf"C:\Users\{username}\.cache\statecraft")
    else:
        raise ValueError(f"Unsupported operating system: {os.name}")
