import pytest
import torch as t
from transformers import AutoTokenizer

from statecraft import StatefulModel

tokeniser = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")


@pytest.fixture
def tensor_input_ids():
    prompt = "Hello, world!"
    return tokeniser(prompt, return_tensors="pt")["input_ids"]


@pytest.mark.parametrize("state", ["koayon/rhyme-state", "koayon/state-a"])
def test_generate(tensor_input_ids, state):
    model = StatefulModel.from_pretrained("state-spaces/mamba-130m-hf", state, device="cpu")

    print("Generation 1")
    out: t.LongTensor = model.generate(tensor_input_ids, max_length=10, num_return_sequences=1, do_sample=False)  # type: ignore
    assert out.shape == (1, 10)
    print(tokeniser.decode(out[0], skip_special_tokens=True))

    # Test that the model outputs change given the state even when greedy decoding

    stateless_model = StatefulModel.from_pretrained("state-spaces/mamba-130m-hf", device="cpu")

    print("Generation 2")
    stateless_out: t.LongTensor = stateless_model.generate(tensor_input_ids, max_length=10, num_return_sequences=1, do_sample=False)  # type: ignore

    assert not t.equal(stateless_out, out)
    print(tokeniser.decode(stateless_out[0], skip_special_tokens=True))
