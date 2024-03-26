import getpass
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import torch as t
from transformers import AutoTokenizer, MambaForCausalLM, PreTrainedModel
from transformers.models.mamba.modeling_mamba import MambaCache, MambaCausalLMOutput

from statecraft.client import StatecraftClient
from statecraft.types import SSMStateMetadata


def get_cached_state(
    saved_state_path: Union[Path, str], cache_dir: Optional[str] = None
) -> tuple[MambaCache, SSMStateMetadata, str]:
    if cache_dir is None:
        cache_dir = StatefulModel._get_default_cache_dir()
    base_path = os.path.join(cache_dir, Path(saved_state_path))

    is_local = os.path.isdir(base_path)
    if not is_local:
        raise ValueError("Path to saved state must be a directory which exists on the system")

    state = t.load(os.path.join(base_path, "state.pt"))

    with open(os.path.join(base_path, "metadata.json"), "r") as json_file:
        metadata_dict = json.load(json_file)

    metadata = SSMStateMetadata(**metadata_dict)

    return state, metadata, base_path


def upload_state(path: Union[Path, str]) -> None:
    state, metadata, base_path = get_cached_state(path)
    print("Uploading state from ", base_path)
    print("Metadata: \n", metadata)

    # Make API call
    raise NotImplementedError
    # StatecraftClient.upload_state(metadata, state)


class StatefulModel(PreTrainedModel):
    def __init__(
        self,
        model: MambaForCausalLM,
        initial_state: Optional[MambaCache],
        device: Optional[str] = None,
    ):
        super().__init__(model.config)
        if initial_state is None:
            initial_state = MambaCache(config=model.config, batch_size=1, device=device)
        self.initial_state: MambaCache = initial_state
        self.model = model

    def forward(
        self,
        input_ids: t.Tensor,
        cache_params: Optional[MambaCache] = None,
        reset_sequence_offset: bool = True,
    ) -> MambaCausalLMOutput:
        # TODO: Deal with change in batch size
        if cache_params is None:
            cache_params = self.initial_state
        if reset_sequence_offset:
            cache_params = self._reset_state_offset(cache_params)

        out: MambaCausalLMOutput = self.model(
            input_ids=input_ids, cache_params=cache_params, use_cache=True
        )
        return out

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        initial_state: Optional[MambaCache] = None,
        device: Optional[str] = None,
    ) -> "StatefulModel":
        model: MambaForCausalLM = MambaForCausalLM.from_pretrained(model_name)  # type: ignore
        stateful_model = cls(model=model, initial_state=initial_state)
        return stateful_model

    def build_state(
        self,
        input_ids: t.Tensor,
        save_state: bool,
        cache_params: Optional[MambaCache] = None,
    ) -> MambaCache:
        out: MambaCausalLMOutput = self.forward(input_ids=input_ids, cache_params=cache_params)
        assert out.cache_params is not None
        if save_state:
            pass
        return out.cache_params

    def save_state(
        self,
        state: MambaCache,
        path: Union[Path, str],
        metadata: SSMStateMetadata,
        cache_dir: Optional[str] = None,
    ) -> None:
        if cache_dir is None:
            cache_dir = self._get_default_cache_dir()
        # Create path to save_location
        base_path = os.path.join(cache_dir, Path(path))
        # Create the cache directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)

        state_path = os.path.join(base_path, "state.pt")
        metadata_path = os.path.join(base_path, "metadata.json")
        # Save state
        t.save(state, state_path)
        print(f"State saved to {state_path}")

        # Save metadata as json file
        with open(metadata_path, "w") as json_file:
            json.dump(metadata.to_dict(), json_file)
        print(f"Metadata saved to {metadata_path}")

    def load_local_state(
        self, saved_state_path: Union[str, Path], cache_dir: Optional[str] = None
    ) -> None:
        state, metadata, base_path = get_cached_state(saved_state_path, cache_dir)

        print("Metadata: \n", metadata)
        self.initial_state = state

        print(f"State loaded from {base_path}")

    @classmethod
    def upload_state(cls, file_location: str) -> None:
        # Make API call
        raise NotImplementedError

    def download_state(self, path: str) -> MambaCache:
        # Make API call
        raise NotImplementedError

    def combine_states(
        self, states: list[MambaCache], weights: Optional[list[float]] = None
    ) -> MambaCache:
        # TODO: Check compatibility

        # Combine states
        if weights is None:
            weights = [1 / len(states)] * len(states)
        raise NotImplementedError

    def update_state(self, state: MambaCache) -> None:
        self.initial_state = state

    @classmethod
    def _get_default_cache_dir(cls) -> str:
        if os.name == "posix":  # For Linux and macOS
            return os.path.expanduser("~/.cache/statecraft/")
        elif os.name == "nt":  # For Windows
            username = getpass.getuser()
            return os.path.expanduser(rf"C:\Users\{username}\.cache\statecraft")
        else:
            raise ValueError(f"Unsupported operating system: {os.name}")

    def _check_state_compatible(self, state: MambaCache) -> bool:
        raise NotImplementedError

    def _reset_state_offset(self, state: MambaCache) -> MambaCache:
        state.seqlen_offset = 0
        return state

    def rag_generate(self, input_str: str) -> str:
        raise NotImplementedError


def main():
    # Set up
    # stateless_model = MambaForCausalLM(MambaConfig())
    stateless_model: MambaForCausalLM = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")  # type: ignore
    tokeniser = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    input_ids: t.Tensor = tokeniser("Hey how are you doing?", return_tensors="pt")[  # type: ignore
        "input_ids"
    ]

    # State generation with no initial state
    model = StatefulModel(
        model=stateless_model,
        initial_state=None,
    )
    generated_state = model.build_state(input_ids=input_ids, save_state=False)
    model.update_state(generated_state)

    # Reuse of state with forward, generating a new state
    out: MambaCausalLMOutput = model(input_ids=input_ids)
    assert out.cache_params is not None
    assert out.logits is not None
    saved_cache_params = out.cache_params

    print(saved_cache_params.ssm_states[1].shape)
    print("ssm_state", saved_cache_params.ssm_states[0])
    print("output", out.logits[0, -1, :])

    # StatefulModel with initial state
    stateful_model = StatefulModel.from_pretrained(
        model_name="state-spaces/mamba-130m-hf",
        initial_state=saved_cache_params,
    )
    print(stateful_model)


def test_saving_state():
    generated_state = model.build_state(input_ids=input_ids, save_state=False)
    model.save_state(
        state=generated_state,
        path="test",
        metadata=SSMStateMetadata(
            state_name="test-state",
            model_name="state_spaces/mamba-130m-hf",
            prompt="Hey how are you doing?",
            description="Test",
        ),
    )


def test_loading_state():
    print(model.initial_state.ssm_states[0].shape)
    print("Previous state", model.initial_state.ssm_states[0][0])
    model.load_local_state(
        saved_state_path="test",
    )
    print(model.initial_state.ssm_states[0].shape)
    print("After state", model.initial_state.ssm_states[0][0])


if __name__ == "__main__":
    tokeniser = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    input_ids: t.Tensor = tokeniser("Hey how are you doing?", return_tensors="pt")[  # type: ignore
        "input_ids"
    ]
    model = StatefulModel.from_pretrained(
        model_name="state-spaces/mamba-130m-hf",
        initial_state=None,
    )

    # test_saving_state()
    test_loading_state()
