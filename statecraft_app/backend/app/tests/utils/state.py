from app import crud
from app.models import State, StateCreate
from app.tests.utils.user import create_random_user
from app.tests.utils.utils import random_lower_string
from sqlmodel import Session


def create_random_state(db: Session) -> State:
    user = create_random_user(db)
    owner_id = user.id
    assert owner_id is not None
    title = random_lower_string()
    description = random_lower_string()
    state_in = StateCreate(title=title, description=description)
    return crud.create_state(session=db, state_in=state_in, owner_id=owner_id)
