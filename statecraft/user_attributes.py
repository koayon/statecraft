from dataclasses import asdict, dataclass


@dataclass
class UserAttributes:
    username: str
    email: str
    token: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
