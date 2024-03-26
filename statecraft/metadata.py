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
