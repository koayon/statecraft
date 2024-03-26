import json
import os
from dataclasses import asdict, dataclass

import typer

from statecraft.utils import get_default_cache_dir


@dataclass
class UserAttributes:
    username: str
    email: str
    token: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def setup():
    cache_dir = get_default_cache_dir()
    username = typer.prompt("Please enter your username")
    email = typer.prompt("Please enter your email address")
    user_attributes = UserAttributes(username=username, email=email, token="")

    # Save user_attributes
    with open(os.path.join(cache_dir, "user_attributes.json"), "w") as json_file:
        json.dump(user_attributes.to_dict(), json_file)

    print("Saved username and email!")


if __name__ == "__main__":
    setup()
