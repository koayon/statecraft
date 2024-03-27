import json
import os
from pathlib import Path
from typing import Union

import requests

from statecraft.metadata import SSMStateMetadata
from statecraft.models import StateOut, StatesOut
from statecraft.user_attributes import UserAttributes
from statecraft.utils import get_default_cache_dir

BASE_URL = "http://www.api.statecrafthub.com"
LOCAL_URL = "http://localhost"


class StatecraftClient:

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.states_url = os.path.join(base_url, "states")

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
        state_short_name = metadata.state_name.split("/")[-1]

        user_attributes = self._fetch_user_attrs()
        username = user_attributes.username

        query_params: dict[str, str] = {
            "state_name": f"{username}/{state_short_name}",
            "model_name": metadata.model_name,
            "prompt": metadata.prompt,
        }
        if metadata.description:
            query_params["description"] = metadata.description
        if metadata.keywords:
            query_params["keywords"] = str(metadata.keywords)

        with open(state_path, "rb") as f:
            state_files = {"state_file": f}

            response = requests.post(self.states_url, params=query_params, files=state_files)
            # TODO: Type this output
        if response.status_code in (200, 201):
            return "State uploaded successfully!"
        else:
            return f"Failed to upload state: {response.status_code}"

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


client = StatecraftClient()


if __name__ == "__main__":
    # state = client.get_state("state-spaces/mamba-130m-hf", "test_user/test_a")
    # print(state)

    states = client.get_states("state-spaces/mamba-130m-hf")
    print(states)

    response_json = client.upload_state(
        SSMStateMetadata(
            model_name="state-spaces/mamba-130m-hf",
            state_name="first-state",
            prompt="This is a test state",
            description="Here's a description...",
        ),
        state_path="/Users/Kola/.cache/statecraft/state-spaces/mamba-130m-hf/CURRENT_USER/test-state/state.pt",
    )

    print(response_json)
