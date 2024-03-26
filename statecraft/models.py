from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class StateOut(BaseModel):
    state_name: str
    description: Optional[str] = None
    model_name: str
    prompt: str
    keywords: Optional[list[str]] = None
    state_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime


StatesOut = list[StateOut]
