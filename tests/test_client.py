from statecraft.core import SSMStateMetadata
from statecraft.utils import get_default_cache_dir

from .client_fixture import client

if __name__ == "__main__":
    # state = client.get_state("state-spaces/mamba-130m-hf", "koayon/state-a")
    # print(state[:100])

    states = client.get_states("state-spaces/mamba-130m-hf")
    print(states)

    cache_dir = get_default_cache_dir()

    response_str = client.upload_state(
        SSMStateMetadata(
            model_name="state-spaces/mamba-130m-hf",
            state_name="first-state",
            prompt="Fee fi fo fum, I smell the blood of an Englishman",
            description="Children's tales",
        ),
        state_path=f"{cache_dir}state-spaces/mamba-130m-hf/CURRENT_USER/test-state/state.pt",  # state_path="/Users/Kola/Documents/VSCode/open_source/statecraft/a.pt",
    )

    print(response_str)
