import torch as t
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM
from transformers.models.mamba.modeling_mamba import MambaCache, MambaCausalLMOutput

if __name__ == "__main__":
    model = MambaForCausalLM(MambaConfig())
    tokeniser = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    input_ids: t.Tensor = tokeniser("Hey how are you doing?", return_tensors="pt")[  # type: ignore
        "input_ids"
    ]

    out: MambaCausalLMOutput = model(input_ids=input_ids, use_cache=True)
    assert out.cache_params is not None
    assert out.logits is not None
    saved_cache_params = out.cache_params
    saved_cache_params.seqlen_offset = 0
    # We probably need a reset_sequence_offset: bool attribute in our function (defaulting to True)

    print(saved_cache_params.ssm_states[1].shape)
    print("ssm_state", saved_cache_params.ssm_states[0])
    print("output", out.logits[0, -1, :])

    # Rerun the model with the saved cache
    out: MambaCausalLMOutput = model(
        input_ids=input_ids, use_cache=True, cache_params=saved_cache_params
    )
    assert out.cache_params is not None
    assert out.logits is not None

    new_cache_params = out.cache_params
    print(new_cache_params.ssm_states[1].shape)
    print("ssm_state", new_cache_params.ssm_states[0])
    print("output", out.logits[0, -1, :])
