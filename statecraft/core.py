import io
import json
import math
import os
from pathlib import Path
from typing import Any, Optional, Union

import torch as t
from einops import einsum
from tqdm import tqdm
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
    """
    Load a cached state from the local cache.

    Parameters
    ----------
    saved_state_name : Union[Path, str]
        The short name of the state, the path where it is saved.
    model_name : str
       The name of the model that the state was created for. This is a Hugging Face model name.
    state_username : str, optional
        Leave blank if the state was built and saved by the current user.
        For states downloaded from the Statecraft Hub, this is the username of the user who uploaded the state.
        by default "CURRENT_USER"
    cache_dir : Optional[str], optional
        The directory where you're storing model states.
        If not provided, the default cache directory is used.
        , by default None

    Returns
    -------
    state: MambaCache
    metadata: SSMStateMetadata
    base_path: str

    Raises
    ------
    CantFindLocalStateError
    CorruptedMetadataError
    ValueError
    """
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

    state = t.load(os.path.join(base_path, "state.pt"), map_location=default_device())

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
    """Upload a state to the [Statecraft Hub](https://www.statecrafthub.com/).
    This will make the state available for others to download and use.

    You should be logged in to your Statecraft account to upload states.

    Thanks for contributing to the community!

    Parameters
    ----------
    path : Union[Path, str]
        The short state_name representing the path to the state you want to upload.
        If you're not sure what the state was called run
        `statecraft.show_available_states(model_name)` to see the available states.
    model_name : str
        The name of the model that the state was created for. This is a Hugging Face model name.

    Raises
    ------
    ValueError
    """
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
    """
    StatefulModels are the core object in Statecraft.
    The StatefulModel class is a wrapper around a Hugging Face model that allows for the loading and saving of states.
    It is designed to be used with models that have a stateful component, such as a SSMs like Mamba.

    The advised way to create a StatefulModel is to use the `from_pretrained` method,
    just like if you were loading a Hugging Face model.

    Args
    ----------
    model : MambaForCausalLM
        The Hugging Face model to wrap.
    initial_state : Optional[MambaCache]
        The initial state of the model. This is the state that the model will start from when generating text.
        If not provided, the model will start from its default state.
    device : Optional[str], optional
        The device to run the model on, by default None
    model_name : Optional[str], optional
        The name of the model. This is a Hugging Face model name.
    """

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

        cache_params = cache_params.to_dtype(self.model.dtype)

        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cache_params=cache_params,
            use_cache=True,
            **kwargs,
        )

        return out

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        initial_state_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> "StatefulModel":
        """Initialise a StatefulModel from a Hugging Face model with an optional initial state.
        You can pass in any additional arguments that you would pass to `AutoModel.from_pretrained`.

        If the either the model or state have been downloaded before they will be collected from the local cache.
        Otherwise, they will be downloaded from the Hugging Face and Statecraft Hubs respectively.

        Parameters
        ----------
        model_name : str
            The name of the model to load. This is a Hugging Face model name.
        initial_state_name : Optional[str], optional
            The short name of the state.
            If the state was built and saved by the current user, you can just write the state_name.
            If the state was downloaded from the Statecraft Hub, you should write the full state_name as it appears on the Statecraft Hub
            e.g. `user_name/state_name`
            , by default None
        device : Optional[str], optional
            The device that the model and state are loaded onto -
            typically `cuda`, `cuda:n`, `cpu` or `mps`
            If not provided, the model will be loaded onto the default device.
            , by default None

        Returns
        -------
        model: StatefulModel
        """
        # Load model from Hugging Face
        model: MambaForCausalLM = MambaForCausalLM.from_pretrained(model_name, device_map=device, **kwargs)  # type: ignore

        # Load initial state
        if initial_state_name is not None:
            initial_state = cls._load_state(model_name, initial_state_name)
        else:
            initial_state = None

        stateful_model = cls(
            model=model, initial_state=initial_state, model_name=model_name, device=device
        )
        return stateful_model.to(device)  # type: ignore

    def build_state(
        self,
        state_name: str,
        prompt: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        cache_params: Optional[MambaCache] = None,
        model_name: Optional[str] = None,
        prompt_reference: Optional[str] = None,
        chunk_size: int = 256,
    ) -> tuple[MambaCache, SSMStateMetadata]:
        """Build a state from a prompt in order to reuse or share it.

        The state is built chunk by chunk so even prompts which are too long to fit in memory can be used.

        Parameters
        ----------
        state_name : str
            Choose a short name for the state, used for the path and for uploads, if you decide to save it.
        prompt : str
            The prompt to generate the state from.
        description : Optional[str], optional
            A description of the state that can be used to understand the context of the state or when to use it, by default None
        tags : Optional[list[str]], optional
            Tags to filter the state by on the Statecraft Hub, by default None
        cache_params : Optional[MambaCache], optional
            An initial state to start building this new state from, by default None
        model_name : Optional[str], optional
            The name of the model that the state was created for. This is a Hugging Face model name.
            If none, the model_name of the StatefulModel is used
            , by default None
        prompt_reference : Optional[str], optional
            For long prompts (over 1000 tokens say), you can save a url as a reference to the prompt here and save that in your metadata rather than the long prompt itself, by default None
        chunk_size : int, optional
            The chunk size used when building the state. You should increase this as your hardware allows, by default 256

        Returns
        -------
        cache_params: MambaCache
        metadata: SSMStateMetadata

        Raises
        ------
        ValueError
        """
        # Check if model_name is provided
        model_name = model_name or self.model_name
        if model_name is None:
            raise ValueError("Model name must be provided.")

        tokeniser = AutoTokenizer.from_pretrained(model_name)
        tokenised_ids: t.Tensor = tokeniser(prompt, return_tensors="pt")["input_ids"]  # type: ignore

        print("Tokenised ids shape: ", tokenised_ids.shape)

        # TODO: Chunk tokenization

        cache_params = cache_params or MambaCache(
            config=self.model.config, batch_size=1, device=self.initial_state.device
        )

        batch, seq_len = tokenised_ids.shape
        num_chunks = math.ceil(seq_len / chunk_size)
        for i in tqdm(range(0, seq_len, chunk_size)):
            # print(f"chunk {i//chunk_size}/{num_chunks} of {num_chunks}")

            chunk = tokenised_ids[:, i : i + chunk_size].to(self.device)

            cache_params = self._build_state(
                input_ids=chunk.to(self.device),
                cache_params=cache_params.to(self.device),
                model_name=model_name,
            )

            # Move the tensors back to CPU and delete them
            chunk = chunk.to("cpu")
            del chunk

            # Clear the GPU cache
            t.cuda.empty_cache()

        metadata = SSMStateMetadata(
            state_name=state_name,
            prompt=prompt_reference or prompt,
            model_name=model_name,
            description=description,
            keywords=tags,
        )

        return cache_params, metadata

    def _build_state(
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

        # print("about to forward")
        with t.no_grad():
            out: MambaCausalLMOutput = self.forward(
                input_ids=input_ids, cache_params=cache_params
            )

        assert out.cache_params is not None

        mamba_cache = MambaCache.from_hf_cache(
            hf_cache=out.cache_params, model_name=model_name
        )
        return mamba_cache

    @classmethod
    def combine_states(
        cls, states: list[MambaCache], weights: Optional[list[float]] = None
    ) -> MambaCache:
        """Combines two or more states into a single state by taking a weighted average of the state tensors.
        Ordinarily, the weights should sum to 1 in this linear combination.

        Parameters
        ----------
        states : list[MambaCache]
            A list of states to combine.
        weights : Optional[list[float]], optional
            Optionally specify weightings for the states.
            If not set, the result will be the mean,
            by default None

        Returns
        -------
        output_cache: MambaCache

        Raises
        ------
        ValueError
        """
        num_states_to_combine = len(states)

        if num_states_to_combine == 1:
            return states[0]
        elif num_states_to_combine == 0:
            raise ValueError("No states provided to combine.")

        original_dtype = states[0].dtype
        batch_size, intermediate_size, ssm_state_size = states[0].ssm_states[0].shape
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
        state_name_path: Optional[Union[Path, str]] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Save the state to disk for reuse.

        Parameters
        ----------
        state : MambaCache
            The state to save.
        metadata : SSMStateMetadata
            The metadata for the state, created when the state was built.
        path : Optional[Union[Path, str]], optional
            A short state name which determines where the state will be saved.
            If not specified we use the state_name in the metadata, by default None
        cache_dir : Optional[str], optional
            The directory where you're storing model states.
            If not provided, the default cache directory is used.
            , by default None
        """
        if cache_dir is None:
            cache_dir = cls._get_default_cache_dir()
        # Create path to save_location
        if state_name_path is None:
            state_name_path = metadata.state_name
        base_path = os.path.join(
            cache_dir, metadata.model_name, "CURRENT_USER", Path(state_name_path)
        )

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
        """Saves the current model state.

        Recommended to instead use `build_state` to save a state from a prompt
        and `save_state` to save a state with the metadata.

        Parameters
        ----------
        state_name : str
            The short name of the state.
        prompt : str
            The prompt used to generate the state from.

        Raises
        ------
        ValueError
        """
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

    def load_state(self, state_name_path: str, cache_dir: Optional[str] = None) -> None:
        """Load a state from disk or from the Statecraft Hub into the model's internal state.


        Parameters
        ----------
        state_name_path : str
            The short name of the state.
            If the state was built and saved by the current user, you can just write the state_name.
            If the state was downloaded from the Statecraft Hub, you should write the full state_name as it appears on the Statecraft Hub,
            e.g. `user_name/state_name`
        cache_dir : Optional[str], optional
            The directory where you're storing model states.
            If not provided, the default cache directory is used.
            , by default None

        Raises
        ------
        ValueError
        """
        if self.model_name is None:
            raise ValueError(
                "Method .load_state(...) requires the model_name to be set.",
                "This happens automatically when using from_pretrained.",
                "Otherwise you can set it manually",
            )
        state = self._load_state(
            model_name=self.model_name, state_name_path=state_name_path, cache_dir=cache_dir
        )
        self.update_state(state)

    def update_state(self, state: MambaCache) -> None:
        """Update the model's internal state to the provided state.

        Parameters
        ----------
        state : MambaCache
            The state to update the model with.
        """
        self.initial_state = state

    def reset_state(self) -> None:
        """Reset the state of the model to the default state."""
        device = self.initial_state.device
        dtype = self.initial_state.dtype
        self.initial_state = MambaCache(
            config=self.model.config, batch_size=1, device=device, dtype=dtype
        )

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

        os.makedirs(base_path, exist_ok=True)

        byte_stream = io.BytesIO(state_bytes)
        try:
            state = t.load(byte_stream, map_location=default_device())
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
