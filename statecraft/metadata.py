from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class SSMStateMetadata:
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
        truncated_description = self._truncate(self.description, 50) if self.description else None
        return f"""SSMStateMetadata(state_name={self.state_name}, model_name={self.model_name},
    prompt={truncated_prompt},
    description={truncated_description},
    keywords={self.keywords}
)"""
