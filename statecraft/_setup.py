import json
import os

import typer

from statecraft.client import client
from statecraft.user_attributes import UserAttributes
from statecraft.utils import get_default_cache_dir


def setup() -> None:
    cache_dir = get_default_cache_dir()
    username = typer.prompt("Please enter your username")
    email = typer.prompt("Please enter your email address")
    user_attributes = UserAttributes(username=username, email=email, token="")

    # Save user_attributes
    with open(os.path.join(cache_dir, "user_attributes.json"), "w") as json_file:
        json.dump(user_attributes.to_dict(), json_file)

    print("Saved username and email!")

    response_code = client.create_user(username, email)
    if response_code in (200, 201):
        print("User created successfully!")
    else:
        print("User creation failed. Please try again.")
