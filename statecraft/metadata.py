from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class SSMStateMetadata:
    """
    The metadata object for a cached state in a Stateful Model.
    This is created when a state is built and is used for understanding the state's context and how to reproduce it.
    The metadata is json serialised when a state is saved to disk.

    Args
    ----------
    state_name : str
        The short name of the state, used for the path and for uploads.
    prompt : str
        The prompt that was used to generate the state. Can also be a prompt_reference, a url which points to a prompt.
    model_name : str
        The name of the model that the state was created for. This is a Hugging Face model name.
    description : str, optional
        A description of the state that can be used to understand the context of the state or when to use it.
    keywords : Optional[list[str]], optional
        Tags to filter the state by on the Statecraft Hub, by default None

    """

    state_name: str
    prompt: str
    model_name: str
    description: Optional[str]
    keywords: Optional[list[str]] = None

    def to_dict(self):
        return asdict(self)

    def _truncate(self, text: str, max_length: int) -> str:
        return text if len(text) <= max_length else text[:max_length] + "..."

    def __repr__(self) -> str:
        truncated_prompt = self._truncate(self.prompt, 50)
        truncated_description = (
            self._truncate(self.description, 50) if self.description else None
        )
        return f"""SSMStateMetadata(state_name={self.state_name}, model_name={self.model_name},
    prompt={truncated_prompt},
    description={truncated_description},
    keywords={self.keywords}
)"""
