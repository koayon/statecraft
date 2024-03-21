from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch as t
from transformers import (
    AutoTokenizer,
    MambaConfig,
    MambaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.mamba.modeling_mamba import MambaCache, MambaCausalLMOutput


@dataclass
class SSMStateMetadata:
    prompt: str
    description: str
    keywords: Optional[list[str]] = None
    model_name: str = "state-spaces/mamba-130m-hf"


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
        self.initial_state = initial_state
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
            cache_params = self.reset_state_offset(cache_params)

        out: MambaCausalLMOutput = self.model(
            input_ids=input_ids, cache_params=cache_params, use_cache=True
        )
        return out

    def build_state(
        self,
        input_ids: t.Tensor,
        save_state: bool,
        cache_params: Optional[MambaCache] = None,
    ) -> MambaCache:
        out: MambaCausalLMOutput = self.forward(input_ids=input_ids, cache_params=cache_params)
        assert out.cache_params is not None
        if save_state:
            self.save_state(out.cache_params)
        return out.cache_params

    def save_state(self, state: MambaCache) -> None:
        raise NotImplementedError

    def upload_state(self, state: MambaCache) -> None:
        raise NotImplementedError

    def load_state(self, path: str) -> None:
        raise NotImplementedError

    def combine_states(self, state1: MambaCache, state2: MambaCache) -> MambaCache:
        # Check compatibility

        # Combine states
        raise NotImplementedError

    def update_state(self, state: MambaCache) -> None:
        self.initial_state = state

    def check_state_compatible(self, state: MambaCache) -> bool:
        raise NotImplementedError

    def reset_state_offset(self, state: MambaCache) -> MambaCache:
        state.seqlen_offset = 0
        return state


if __name__ == "__main__":
    # Set up
    stateless_model = MambaForCausalLM(MambaConfig())
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
