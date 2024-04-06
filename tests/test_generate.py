import pytest
import torch as t
from transformers import AutoTokenizer

from statecraft import StatefulModel

tokeniser = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")


@pytest.fixture
def tensor_input_ids():
    prompt = "Hello, world!"
    return tokeniser(prompt, return_tensors="pt")["input_ids"]


def test_generate(tensor_input_ids):
    model = StatefulModel.from_pretrained("state-spaces/mamba-130m-hf", device="cpu")
    out: t.LongTensor = model.generate(tensor_input_ids, max_length=10, num_return_sequences=1)  # type: ignore
    assert out.shape == (1, 10)
