import io
import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import torch as t
from einops import einsum
from transformers import AutoTokenizer, MambaForCausalLM, PreTrainedModel
from transformers.generation.utils import GenerateOutput
from transformers.models.mamba.modeling_mamba import MambaCache as HFMambaCache
from transformers.models.mamba.modeling_mamba import MambaCausalLMOutput

from statecraft.cache import MambaCache
from statecraft.client import client
from statecraft.metadata import SSMStateMetadata
from statecraft.utils import default_device, get_default_cache_dir


class CantFindLocalStateError(Exception):
    pass


class CorruptedMetadataError(Exception):
    pass


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
        raise CantFindLocalStateError(
            "Path to saved state must be a directory which exists on the system"
        )

    state = t.load(os.path.join(base_path, "state.pt"))

    with open(os.path.join(base_path, "metadata.json"), "r") as json_file:
        metadata_dict: dict[str, Any] = json.load(json_file)

    for key in ("state_name", "model_name", "prompt"):
        if key not in metadata_dict:
            raise CorruptedMetadataError(f"Metadata must contain key: {key}")

    metadata = SSMStateMetadata(
        state_name=metadata_dict.get("state_name"),  # type: ignore
        model_name=metadata_dict.get("model_name"),  # type: ignore
        prompt=metadata_dict.get("prompt"),  # type: ignore
        description=metadata_dict.get("description"),
        keywords=metadata_dict.get("keywords"),
    )

    print("Metadata: \n", metadata)
    print(f"State loaded from {base_path}")

    if not isinstance(state, MambaCache) and isinstance(state, HFMambaCache):
        state = MambaCache.from_hf_cache(hf_cache=state, model_name=model_name)
    elif isinstance(state, bytes):
        raise ValueError("State is not a MambaCache object.")

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

        if device is not None:
            initial_state = initial_state.to(device)

        self.initial_state: MambaCache = initial_state
        torch_device = default_device() if device is None else t.device(device)
        self.model: MambaForCausalLM = model.to(device=torch_device)  # type: ignore
        self.model_name = model_name

    def forward(
        self,
        input_ids: t.Tensor,
        cache_params: Optional[MambaCache] = None,
        reset_sequence_offset: bool = True,
        **kwargs,
    ) -> MambaCausalLMOutput:
        # TODO: Deal with change in batch size
        if cache_params is None:
            cache_params = self.initial_state
        if reset_sequence_offset:
            cache_params = self._reset_state_offset(cache_params)

        out: MambaCausalLMOutput = self.model(
            input_ids=input_ids, cache_params=cache_params, use_cache=True, **kwargs
        )
        return out

    def generate(
        self,
        input_ids: t.Tensor,
        max_length: int = 50,
        cache_params: Optional[MambaCache] = None,
        reset_sequence_offset: bool = True,
        **kwargs,
    ) -> Union[t.LongTensor, GenerateOutput]:
        if cache_params is None:
            cache_params = self.initial_state
        if reset_sequence_offset:
            cache_params = self._reset_state_offset(cache_params)

        out = self.model.generate(
            input_ids=input_ids, max_length=max_length, cache_params=cache_params, **kwargs
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
        model: MambaForCausalLM = MambaForCausalLM.from_pretrained(model_name, device_map=device)  # type: ignore

        # Load initial state
        if initial_state_name is not None:
            initial_state = cls._load_state(model_name, initial_state_name)
        else:
            initial_state = None

        stateful_model = cls(
            model=model, initial_state=initial_state, model_name=model_name, device=device
        )
        return stateful_model

    def build_state(
        self,
        input_ids: t.Tensor,
        cache_params: Optional[MambaCache] = None,
        model_name: Optional[str] = None,
    ) -> MambaCache:
        # Check if model_name is provided
        if model_name is None and self.model_name is None:
            raise ValueError("Model name must be provided.")
        elif model_name is None:
            model_name = self.model_name
        assert model_name is not None

        out: MambaCausalLMOutput = self.forward(input_ids=input_ids, cache_params=cache_params)
        assert out.cache_params is not None

        mamba_cache = MambaCache.from_hf_cache(
            hf_cache=out.cache_params, model_name=model_name
        )
        return mamba_cache

    @classmethod
    def combine_states(
        cls, states: list[MambaCache], weights: Optional[list[float]] = None
    ) -> MambaCache:
        num_states_to_combine = len(states)

        if num_states_to_combine == 1:
            return states[0]
        elif num_states_to_combine == 0:
            raise ValueError("No states provided to combine.")

        original_dtype = states[0].dtype
        batch_size, intemediate_size, ssm_state_size = states[0].ssm_states[0].shape
        num_layers = len(states[0].conv_states)

        # Check if all states are compatible
        for i, state in enumerate(states):
            if i == 0:
                pass
            else:
                cls._check_state_compatible(states[0], state)

        # Build weights tensor
        if weights is None:
            weights = [1 / len(states)] * len(states)
        elif len(weights) != len(states):
            raise ValueError(
                f"Number of weights provided ({len(weights)}) must match the number of states provided ({len(states)})"
            )
        weights_tensor = t.tensor(weights)

        # Combine Conv states
        conv_states_stacked = [
            t.stack(list(states[i].conv_states.values())) for i in range(num_states_to_combine)
        ]  # state_num list[layer batch intermediate_size conv_kernel_size]

        tensor_conv_states_stacked = t.stack(
            conv_states_stacked
        )  # state_num layer batch intermediate_size conv_kernel_size

        new_conv_states = einsum(
            weights_tensor,
            tensor_conv_states_stacked.to(dtype=t.float32),
            "state_num, state_num layer batch intermediate_size conv_kernel_size -> layer batch intermediate_size conv_kernel_size",
        )

        new_conv_states_dict = {
            i: new_conv_states[i].to(dtype=original_dtype) for i in range(num_layers)
        }

        # Combine SSM states
        ssm_states_stacked = [
            t.stack(list(states[i].ssm_states.values())) for i in range(num_states_to_combine)
        ]  # state_num list[layer batch intermediate_size ssm_state_size]

        tensor_ssm_states_stacked = t.stack(
            ssm_states_stacked
        )  # state_num layer batch intermediate_size ssm_state_size

        new_ssm_states = einsum(
            weights_tensor,
            tensor_ssm_states_stacked.to(dtype=t.float32),
            "state_num, state_num layer batch intermediate_size ssm_state_size -> layer batch intermediate_size ssm_state_size",
        )

        new_ssm_states_dict = {
            i: new_ssm_states[i].to(dtype=original_dtype) for i in range(num_layers)
        }

        out = states[0]
        out.conv_states = new_conv_states_dict
        out.ssm_states = new_ssm_states_dict

        return out

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
        device = self.initial_state.device
        self.initial_state = MambaCache(config=self.model.config, batch_size=1, device=device)

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

        byte_stream = io.BytesIO(state_bytes)
        try:
            state = t.load(byte_stream)
        except Exception as e:
            raise IOError(
                f"Failed to parse downloaded state from bytes.",
            )

        state_path = os.path.join(base_path, "state.pt")
        t.save(state, state_path)

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
        except CantFindLocalStateError as e:
            print(
                f"State {state_name_path} not found in local cache. Trying to load from server."
            )
            pass
        except CorruptedMetadataError as e:
            raise ValueError(
                f"Metadata for state {state_name_path} is corrupted. Try deleting the file and running again."
            )
        except Exception as e:
            print(e)
            pass

        try:  # Try loading from server
            state_bytes = client.get_state(model_name, state_name_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load state from local cache and server.",
                f"Model name: {model_name}",
                f"State name: {state_name_path}",
                str(e),
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

    @classmethod
    def _check_state_compatible(cls, state1: MambaCache, state2: MambaCache) -> bool:
        if state1.dtype != state2.dtype:
            raise ValueError(
                f"""States must have the same dtype.
The first state provided has the dtype {state1.dtype}, which is different from {state2.dtype}, the dtype of the second state provided."""
            )

        if state1.conv_states.keys() != state2.conv_states.keys():
            raise ValueError(
                f"""States must have the same convolutional states.
The first state provided has the keys {state1.conv_states.keys()}, which are different from {state2.conv_states.keys()}, the keys of the second state provided."""
            )

        if state1.ssm_states.keys() != state2.ssm_states.keys():
            raise ValueError(
                f"""States must have the same SSM states.
The first state provided has the keys {state1.ssm_states.keys()}, which are different from {state2.ssm_states.keys()}, the keys of the second state provided."""
            )

        if state1.ssm_states[0].shape != state2.ssm_states[0].shape:
            raise ValueError(f"""SSM States should be the same shape.""")

        return True

    def _reset_state_offset(self, state: MambaCache) -> MambaCache:
        state.seqlen_offset = 0
        return state
