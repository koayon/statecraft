import json
from pathlib import Path
from typing import Any, Optional, Union

import requests

from statecraft.types import SSMStateMetadata


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
            print("Got state: ", raw_bytes[:100])
            return raw_bytes
        else:
            raise ValueError(f"Failed to get state: {response.text}")

    # @classmethod
    # def get_state_metadata(cls, state_full_identifier: str) -> SSMStateMetadata:
    #     response = requests.get(f"metadata/{cls.states_url}/{state_full_identifier}")
    #     if response.status_code == 200:
    #         metadata_dict = response.json()
    #         metadata = SSMStateMetadata(**metadata_dict)
    #         return metadata
    #     else:
    #         raise ValueError(f"Failed to get state metadata: {response.text}")

    @classmethod
    def upload_state(
        cls,
        metadata: SSMStateMetadata,
        state_path: Union[Path, str],
    ) -> dict:
        state_short_name = metadata.state_name
        username = "test_user"
        query_params: dict[str, str] = {
            "state_name": f"{username}/{state_short_name}",
            "model_name": metadata.model_name,
            "prompt": metadata.prompt,
        }
        if metadata.description:
            query_params["description"] = metadata.description
        # if metadata.keywords:
        #     query_params["keywords"] = str(metadata.keywords)

        with open(state_path, "rb") as f:
            state_files = {"state_file": f}

            response = requests.post(cls.states_url, params=query_params, files=state_files)
        return response.json()


if __name__ == "__main__":
    client = StatecraftClient()
    state = client.get_state("state-spaces/mamba-130m-hf", "test_user/test_a")
    print(state)

    # response_json = client.upload_state(
    #     SSMStateMetadata(
    #         model_name="state-spaces/mamba-130m-hf",
    #         state_name="test-state",
    #         prompt="This is a test state",
    #         description="This is a test description",
    #     ),
    #     state_path="/Users/Kola/.cache/statecraft/state_spaces/mamba-130m-hf/CURRENT_USER/test-state/state.pt",
    # )

    # print(response_json)
