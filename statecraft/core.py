import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import torch as t
from transformers import AutoTokenizer, MambaForCausalLM, PreTrainedModel
from transformers.models.mamba.modeling_mamba import MambaCache, MambaCausalLMOutput

from statecraft.client import client
from statecraft.metadata import SSMStateMetadata
from statecraft.utils import get_default_cache_dir


def get_cached_state(
    saved_state_name: Union[Path, str],
    model_name: str,
    state_username: str = "CURRENT_USER",
    cache_dir: Optional[str] = None,
) -> tuple[MambaCache, SSMStateMetadata, str]:
    if cache_dir is None:
        cache_dir = StatefulModel._get_default_cache_dir()

    if "/" in str(saved_state_name):
        base_path = os.path.join(cache_dir, model_name, Path(saved_state_name))
    else:
        base_path = os.path.join(cache_dir, model_name, state_username, Path(saved_state_name))

    is_local = os.path.isdir(base_path)
    if not is_local:
        raise ValueError("Path to saved state must be a directory which exists on the system")

    state = t.load(os.path.join(base_path, "state.pt"))

    with open(os.path.join(base_path, "metadata.json"), "r") as json_file:
        metadata_dict: dict[str, Any] = json.load(json_file)

    for key in ("state_name", "model_name", "prompt"):
        if key not in metadata_dict:
            raise ValueError(f"Metadata must contain key: {key}")

    metadata = SSMStateMetadata(
        state_name=metadata_dict.get("state_name"),  # type: ignore
        model_name=metadata_dict.get("model_name"),  # type: ignore
        prompt=metadata_dict.get("prompt"),  # type: ignore
        description=metadata_dict.get("description"),
        keywords=metadata_dict.get("keywords"),
    )

    print("Metadata: \n", metadata)
    print(f"State loaded from {base_path}")

    return state, metadata, base_path


def upload_state(path: Union[Path, str], model_name: str) -> None:
    _state, metadata, base_path = get_cached_state(path, model_name)

    if metadata.model_name != model_name:
        raise ValueError(
            "Model name in metadata does not match the model name provided.",
            f"Model name in metadata: {metadata.model_name}",
            f"Model name provided: {model_name}",
        )

    print("Uploading state from ", base_path)
    print("Metadata: \n", metadata)

    state_path = os.path.join(base_path, "state.pt")

    # Make API call
    client.upload_state(metadata, state_path)


class StatefulModel(PreTrainedModel):
    def __init__(
        self,
        model: MambaForCausalLM,
        initial_state: Optional[MambaCache],
        device: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        super().__init__(model.config)
        if initial_state is None:
            initial_state = MambaCache(config=model.config, batch_size=1, device=device)
        self.initial_state: MambaCache = initial_state
        self.model = model
        self.model_name = model_name

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
        initial_state_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "StatefulModel":
        # Load model from Hugging Face
        model: MambaForCausalLM = MambaForCausalLM.from_pretrained(model_name)  # type: ignore

        # Load initial state
        if initial_state_name is not None:
            initial_state = cls._load_state(model_name, initial_state_name)
        else:
            initial_state = None

        stateful_model = cls(model=model, initial_state=initial_state, model_name=model_name)
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

    def rag_generate(self, input_str: str) -> str:
        raise NotImplementedError

    def combine_states(
        self, states: list[MambaCache], weights: Optional[list[float]] = None
    ) -> MambaCache:
        # TODO: Check compatibility

        # Combine states
        if weights is None:
            weights = [1 / len(states)] * len(states)
        raise NotImplementedError

    # STATE SAVING
    @classmethod
    def save_state(
        cls,
        state: MambaCache,
        metadata: SSMStateMetadata,
        path: Optional[Union[Path, str]] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        if cache_dir is None:
            cache_dir = cls._get_default_cache_dir()
        # Create path to save_location
        if path is None:
            path = metadata.state_name
        base_path = os.path.join(cache_dir, metadata.model_name, "CURRENT_USER", Path(path))

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

    def save_current_state(self, state_name: str, prompt: str) -> None:
        if self.model_name is None:
            raise ValueError(
                "Method .save_current_state(...) requires the model_name to be set.",
                "This happens automatically when using from_pretrained.",
                "Otherwise you can set it manually or save the state using .save_state(...)",
            )
        metadata = SSMStateMetadata(
            state_name=state_name,
            prompt=prompt,
            model_name=self.model_name,
            description="",
        )
        self.save_state(state=self.initial_state, metadata=metadata)

    # STATE LOADING

    def load_state(self, path: str, cache_dir: Optional[str] = None) -> None:
        if self.model_name is None:
            raise ValueError(
                "Method .load_state(...) requires the model_name to be set.",
                "This happens automatically when using from_pretrained.",
                "Otherwise you can set it manually",
            )
        state = self._load_state(
            model_name=self.model_name, state_name_path=path, cache_dir=cache_dir
        )
        self.update_state(state)

    def update_state(self, state: MambaCache) -> None:
        self.initial_state = state

    def reset_state(self) -> None:
        self.initial_state = MambaCache(config=self.model.config, batch_size=1, device=None)

    # HELPER METHODS

    @classmethod
    def _save_state_binaries(
        cls,
        state_bytes: bytes,
        model_name_path: str,
        state_name_path: str,
        cache_dir: Optional[str] = None,
    ) -> None:
        cache_dir = cls._get_default_cache_dir() if cache_dir is None else cache_dir
        base_path = os.path.join(cache_dir, model_name_path, state_name_path)

        os.makedirs(base_path, exist_ok=False)

        state_path = os.path.join(base_path, "state.pt")
        with open(state_path, "wb") as f:
            f.write(state_bytes)

        metadata_path = os.path.join(base_path, "metadata.json")
        metadata = SSMStateMetadata(
            state_name=state_name_path,
            model_name=model_name_path,
            prompt="See statecrafthub.com for details on the prompt used",
            description=None,
        )

        with open(metadata_path, "w") as json_file:
            json.dump(metadata.to_dict(), json_file)

    @classmethod
    def _load_state(
        cls, model_name: str, state_name_path: str, cache_dir: Optional[str] = None
    ) -> MambaCache:
        try:  # Try loading from local cache
            state, _metadata, _base_path = get_cached_state(
                saved_state_name=state_name_path,
                model_name=model_name,
                cache_dir=cache_dir,
            )
            return state
        except Exception as e:
            pass

        try:  # Try loading from server
            state_bytes = client.get_state(model_name, state_name_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load state from local cache and server.",
                f"Model name: {model_name}",
                f"State name: {state_name_path}",
                e,
            )

        try:
            cls._save_state_binaries(
                state_bytes=state_bytes,
                model_name_path=model_name,
                state_name_path=state_name_path,
                cache_dir=cache_dir,
            )
        except Exception as e:
            raise IOError(
                f"Failed to save state binaries to local cache.",
                f"Model name: {model_name}",
                f"State name: {state_name_path}",
                e,
            )

        state, _metadata, _base_path = get_cached_state(
            saved_state_name=state_name_path,
            model_name=model_name,
            cache_dir=cache_dir,
        )
        return state

    @classmethod
    def _get_default_cache_dir(cls) -> str:
        return get_default_cache_dir()

    def _check_state_compatible(self, state: MambaCache) -> bool:
        raise NotImplementedError

    def _reset_state_offset(self, state: MambaCache) -> MambaCache:
        state.seqlen_offset = 0
        return state
