import pytest
import torch as t
from transformers.models.mamba.modeling_mamba import MambaCache, MambaConfig

from statecraft import StatefulModel

small_model_config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
medium_model_config = MambaConfig.from_pretrained("state-spaces/mamba-370m-hf")


class Test_CheckStateCompatible:

    # Given two MambaCache objects with the same dtype, same convolutional states, and same SSM states, the method should return True.
    def test_same_states(self):
        state1 = MambaCache(config=small_model_config, batch_size=1, device=None)
        state2 = MambaCache(config=small_model_config, batch_size=1, device=None)

        result = StatefulModel._check_state_compatible(state1, state2)

        assert result == True

    # Given two MambaCache objects with different dtypes, the method should raise a ValueError with a message indicating that the states must have the same dtype.
    def test_different_dtypes(self):
        state1 = MambaCache(config=small_model_config, batch_size=1, device=None)
        state2 = MambaCache(config=small_model_config, batch_size=1, device=None)

        state1.dtype = t.float32
        state2.dtype = t.float16

        with pytest.raises(ValueError) as e:
            StatefulModel._check_state_compatible(state1, state2)

    # Given two MambaCache objects with differently sized states, the method should raise a ValueError with a message indicating that the states must have the same convolutional states.
    def test_different_conv_states(self):
        state1 = MambaCache(config=small_model_config, batch_size=1, device=None)
        state2 = MambaCache(config=medium_model_config, batch_size=1, device=None)

        with pytest.raises(ValueError) as e:
            StatefulModel._check_state_compatible(state1, state2)


class TestCombineStates:

    # Combining two compatible states with equal weights returns a new state with correct conv and ssm states
    def test_combine_states_equal_weights(self):
        states = [
            MambaCache(config=small_model_config, batch_size=1),
            MambaCache(config=small_model_config, batch_size=1),
        ]

        states[0].ssm_states[2] = t.randn(states[0].ssm_states[2].shape)
        states[1].ssm_states[2] = t.randn(states[1].ssm_states[2].shape)

        expected_output = (0.5 * states[0].ssm_states[2] + 0.5 * states[1].ssm_states[2]).to(
            dtype=states[0].ssm_states[2].dtype
        )

        combined_state = StatefulModel.combine_states(states)

        assert isinstance(combined_state, MambaCache)
        assert len(combined_state.conv_states) == len(states[0].conv_states)
        assert len(combined_state.ssm_states) == len(states[0].ssm_states)
        assert t.norm(combined_state.ssm_states[2] - expected_output) < 0.1

    # Combining an empty list of states raises a ValueError
    def test_combine_states_empty_list(self):
        with pytest.raises(ValueError):
            StatefulModel.combine_states([])

    # Combining multiple compatible states with different weights returns a new state with correct conv and ssm states
    def test_combine_states_different_weights(self):
        states = [
            MambaCache(config=small_model_config, batch_size=1),
            MambaCache(config=small_model_config, batch_size=1),
        ]

        states[0].ssm_states[2] = t.randn(states[0].ssm_states[2].shape)
        states[1].ssm_states[2] = t.randn(states[1].ssm_states[2].shape)

        weights = [0.3, 0.7]

        expected_output = (0.3 * states[0].ssm_states[2] + 0.7 * states[1].ssm_states[2]).to(
            dtype=states[0].ssm_states[2].dtype
        )

        combined_state = StatefulModel.combine_states(states, weights)

        assert isinstance(combined_state, MambaCache)
        assert t.norm(combined_state.ssm_states[2] - expected_output) < 0.1
