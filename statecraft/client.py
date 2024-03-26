import json
from pathlib import Path
from typing import Optional, Union

import requests

from statecraft.types import SSMStateMetadata


class StatecraftClient:
    base_url = "http://localhost"
    states_url = f"{base_url}/states"

    def __init__(self):
        pass

    @classmethod
    def get_state(cls, state_full_identifier: str) -> dict:
        response = requests.get(f"{cls.states_url}/{state_full_identifier}")
        return response.json()

    @classmethod
    def upload_state(
        cls,
        metadata: SSMStateMetadata,
        state: bytes,
    ) -> dict:
        response = requests.post(cls.states_url, json=state)
        return response.json()


if __name__ == "__main__":
    client = StatecraftClient()
    state = client.get_state("state-spaces/mamba-130m-hf/test_user/test_short_state_name")
    print(state)

#     "state_name": "string",
#   "description": "string",
#   "model_name": "string",
#   "prompt": "string",
#   "keywords": [
#     "string"
#   ],
#   "state": "string"
