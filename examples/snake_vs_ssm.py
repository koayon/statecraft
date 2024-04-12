import torch as t
from transformers import AutoTokenizer

from examples.prompts import MAMBA_ABSTRACT, MAMBA_WIKI
from statecraft import StatefulModel, upload_state

MODEL_NAME: str = "state-spaces/mamba-2.8b-hf"


def build_and_generate(prompt: str, state_name: str):
    state, metadata = model.build_state(prompt=prompt, state_name=state_name)
    print(f"{state_name} state built")

    # model.save_state(state, metadata=metadata)
    # upload_state(path=state_name, model_name=MODEL_NAME)

    # Generate Mamba SSM output
    abstract_output_ids = model.generate(
        question_input_ids.to(model.device), cache_params=state
    )
    print(tokeniser.decode(abstract_output_ids[0], skip_special_tokens=True))
    print(metadata)

    model.reset_state()


if __name__ == "__main__":
    model = StatefulModel.from_pretrained(model_name=MODEL_NAME, device="cpu")
    tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

    question_input_ids: t.Tensor = tokeniser("In summary,", return_tensors="pt")[
        "input_ids"
    ]  # type: ignore

    # Build up the two states

    build_and_generate(
        prompt=MAMBA_ABSTRACT,
        state_name="mamba_abstract_state",
    )

    build_and_generate(
        prompt=MAMBA_WIKI,
        state_name="mamba_wiki_state",
    )
