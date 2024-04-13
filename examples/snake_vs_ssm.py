import torch as t
from transformers import AutoTokenizer

from examples.prompts import MAMBA_PAPER, MAMBA_WIKI
from statecraft import StatefulModel, upload_state

MODEL_NAME: str = "state-spaces/mamba-2.8b-hf"


def build_and_generate(prompt: str, state_name: str):
    state, metadata = model.build_state(prompt=prompt, state_name=state_name)
    print(f"{state_name} state built")

    # model.save_state(state, metadata=metadata)
    # upload_state(path=state_name, model_name=MODEL_NAME)

    print()

    state.seqlen_offset = 1

    model.update_state(state)

    # Generate Mamba SSM output
    output_ids = model.generate(
        question_input_ids.to(model.device), reset_sequence_offset = False, max_length=10, num_return_sequences=1, do_sample=False)

    print(tokeniser.decode(output_ids[0], skip_special_tokens=True))
    print(metadata)

    model.reset_state()


if __name__ == "__main__":
    model = StatefulModel.from_pretrained(model_name=MODEL_NAME)
    model = model.to(t.device("cuda"))
    tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("memory allocated",  t.cuda.memory_allocated()/1024//1024, "MB")

    question_input_ids: t.Tensor = tokeniser("In summary,", return_tensors="pt")[
        "input_ids"
    ]  # type: ignore

    # Build up the two states
    build_and_generate(
        prompt=MAMBA_PAPER,
        state_name="mamba_abstract_state",
    )

    build_and_generate(
        prompt=MAMBA_WIKI,
        state_name="mamba_wiki_state",
    )
