import copy
from typing import Optional

import torch as t
from transformers.models.mamba.modeling_mamba import MambaCache as HFMambaCache
from transformers.models.mamba.modeling_mamba import MambaConfig


class MambaCache(HFMambaCache):
    def to(self, device):
        mamba_cache = copy.deepcopy(self)

        ssm_states = {
            layer_num: layer_cache.to(device)
            for layer_num, layer_cache in mamba_cache.ssm_states.items()
        }

        conv_states = {
            layer_num: layer_cache.to(device)
            for layer_num, layer_cache in mamba_cache.conv_states.items()
        }

        mamba_cache.ssm_states = ssm_states
        mamba_cache.conv_states = conv_states

        return mamba_cache

    @property
    def device(self):
        return self.ssm_states[0].device

    # def __iadd__(self, other: "MambaCache"):
    #     for layer_num, layer_cache in self.ssm_states.items():
    #         layer_cache += other.ssm_states[layer_num]
    #     for layer_num, layer_cache in self.conv_states.items():
    #         layer_cache += other.conv_states[layer_num]
    #     return self

    @classmethod
    def from_hf_cache(
        cls, hf_cache: HFMambaCache, model_name: str, device: Optional[str] = None
    ) -> "MambaCache":
        mamba_config = MambaConfig.from_pretrained(model_name)

        batch_size = hf_cache.ssm_states[0].shape[0]

        # Initialise the Mamba cache
        mamba_cache = cls(config=mamba_config, batch_size=batch_size, device=device)

        # Copy the states from the HF cache to the Mamba cache
        mamba_cache.conv_states = hf_cache.conv_states
        mamba_cache.ssm_states = hf_cache.ssm_states
        mamba_cache.seqlen_offset = hf_cache.seqlen_offset
        mamba_cache.dtype = hf_cache.dtype

        return mamba_cache
