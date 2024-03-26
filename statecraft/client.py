import json
import os
from pathlib import Path
from typing import Union

import requests

from statecraft.metadata import SSMStateMetadata
from statecraft.models import StateOut, StatesOut
from statecraft.user_attributes import UserAttributes
from statecraft.utils import get_default_cache_dir


class StatecraftClient:
    base_url = "http://localhost"
    states_url = f"{base_url}/states"

    def __init__(self):
        pass

    @classmethod
    def get_state(cls, model_name: str, state_name: str) -> bytes:
        response = requests.get(f"{cls.states_url}/{model_name}/{state_name}")
        if response.status_code == 200:
            raw_bytes = response.content
            print("Got state - raw_bytes: ", raw_bytes[:100])
            return raw_bytes
        else:
            raise ValueError(f"Failed to get state: {response.text}")

    @classmethod
    def upload_state(
        cls,
        metadata: SSMStateMetadata,
        state_path: Union[Path, str],
    ) -> dict:
        state_short_name = metadata.state_name.split("/")[-1]

        user_attributes = cls._fetch_user_attrs()
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

            response = requests.post(cls.states_url, params=query_params, files=state_files)
            # TODO: Type this output
        return response.json()

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

    @classmethod
    def get_states(cls, model_name: str) -> list[str]:
        model_user_name, model_short_name = model_name.split("/")
        query_params = {
            "model_user_name": model_user_name,
            "model_short_name": model_short_name,
        }

        response = requests.get(
            cls.states_url,
            params=query_params,
        )
        parsed_response: list[dict[str, str]] = response.json()  # StatesOut

        states_list = [
            f'{response_state_dict["model_name"]}/{response_state_dict["state_name"]}'
            for response_state_dict in parsed_response
        ]

        return states_list


if __name__ == "__main__":
    client = StatecraftClient()
    state = client.get_state("state-spaces/mamba-130m-hf", "test_user/test_a")
    print(state)

    states = client.get_states("state-spaces/mamba-130m-hf")
    print(states)

    # response_json = client.upload_state(
    #     SSMStateMetadata(
    #         model_name="state-spaces/mamba-130m-hf",
    #         state_name="test-state",
    #         prompt="This is a test state",
    #         description="This is a test description",
    #     ),
    #     state_path="/Users/Kola/.cache/statecraft/state-spaces/mamba-130m-hf/CURRENT_USER/test-state/state.pt",
    # )

    # print(response_json)
