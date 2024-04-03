import json
import os
from pathlib import Path
from typing import Union

import requests

from statecraft.metadata import SSMStateMetadata
from statecraft.models import StateOut, StatesOut
from statecraft.user_attributes import UserAttributes
from statecraft.utils import get_default_cache_dir

BASE_URL = "https://www.api.statecrafthub.com/"
LOCAL_URL = "http://localhost/"


class StatecraftClient:

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.states_url = f"{base_url}states/"

    def get_state(self, model_name: str, state_name: str) -> bytes:
        full_url = os.path.join(self.states_url, model_name, state_name)

        response = requests.get(full_url)

        if response.status_code == 200:
            raw_bytes = response.content
            print("Got state - raw_bytes: ", raw_bytes[:100])
            return raw_bytes
        else:
            raise ValueError(f"Failed to get state: {response.text}")

    def upload_state(
        self,
        metadata: SSMStateMetadata,
        state_path: Union[Path, str],
    ) -> str:
        url = self.states_url

        print(url)

        state_short_name = metadata.state_name.split("/")[-1]

        user_attributes = self._fetch_user_attrs()
        username = user_attributes.username

        query_params = {
            "state_name": f"{username}/{state_short_name}",
            "model_name": metadata.model_name,
            "prompt": metadata.prompt,
        }

        if metadata.description:
            query_params["description"] = metadata.description
        if metadata.keywords:
            query_params["keywords"] = str(metadata.keywords)

        files = {
            "state_file": (
                f"{state_short_name}.pt",
                open(state_path, "rb"),
                "application/octet-stream",
            )
        }

        headers = {
            "accept": "application/json",
        }
        response = requests.post(url, params=query_params, files=files, headers=headers)

        try:
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

        if response.status_code in (200, 201):
            print(response.text)
            return "State uploaded successfully!"
        else:
            return f"Failed to upload state: {response.status_code}"

    def get_states(self, model_name: str) -> list[str]:
        model_user_name, model_short_name = model_name.split("/")
        query_params = {
            "model_user_name": model_user_name,
            "model_short_name": model_short_name,
        }

        response = requests.get(
            self.states_url,
            params=query_params,
        )
        print(response.text)
        parsed_response: list[dict[str, str]] = response.json()  # StatesOut

        states_list = [
            f'{response_state_dict["model_name"]}/{response_state_dict["state_name"]}'
            for response_state_dict in parsed_response
        ]

        return states_list

    def create_user(self, username: str, email: str) -> int:
        response = requests.post(
            os.path.join(self.base_url, "users"),
            json={"username": username, "email": email},
        )
        response_code = response.status_code
        return response_code

    @classmethod
    def _fetch_user_attrs(cls) -> UserAttributes:
        cache_dir = get_default_cache_dir()

        # Check if user attributes are saved
        if os.path.exists(os.path.join(cache_dir, "user_attributes.json")):
            with open(os.path.join(cache_dir, "user_attributes.json"), "r") as f:
                user_attributes_dict: dict = json.load(f)
                user_attributes = UserAttributes(
                    username=user_attributes_dict["username"],
                    email=user_attributes_dict["email"],
                    token=user_attributes_dict["token"],
                )
                return user_attributes
        else:
            raise ValueError("User attributes not found. Please run statecraft.setup()")


client = StatecraftClient()
