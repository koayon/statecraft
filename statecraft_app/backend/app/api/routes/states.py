from typing import Any

from app.api.deps import CurrentUser, SessionDep
from app.models import Message, State, StateCreate, StateOut, StatesOut, StateUpdate
from fastapi import APIRouter, HTTPException
from sqlmodel import func, select

router = APIRouter()


@router.get("/", response_model=StatesOut)
def read_states(
    session: SessionDep, current_user: CurrentUser, skip: int = 0, limit: int = 100
) -> Any:
    """
    Retrieve states.
    """

    if current_user.is_superuser:
        statement = select(func.count()).select_from(State)
        count = session.exec(statement).one()
        statement = select(State).offset(skip).limit(limit)
        states = session.exec(statement).all()
    else:
        statement = (
            select(func.count()).select_from(State).where(State.owner_id == current_user.id)
        )
        count = session.exec(statement).one()
        statement = (
            select(State).where(State.owner_id == current_user.id).offset(skip).limit(limit)
        )
        states = session.exec(statement).all()

    return StatesOut(data=states, count=count)


@router.get("/{id}", response_model=StateOut)
def read_state(session: SessionDep, current_user: CurrentUser, id: int) -> Any:
    """
    Get state by ID.
    """
    state = session.get(State, id)
    if not state:
        raise HTTPException(status_code=404, detail="State not found")
    if not current_user.is_superuser and (state.owner_id != current_user.id):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    return state


@router.post("/", response_model=StateOut)
def create_state(
    *, session: SessionDep, current_user: CurrentUser, state_in: StateCreate
) -> Any:
    """
    Create new state.
    """
    state = State.model_validate(state_in, update={"owner_id": current_user.id})
    session.add(state)
    session.commit()
    session.refresh(state)
    return state


@router.put("/{id}", response_model=StateOut)
def update_state(
    *, session: SessionDep, current_user: CurrentUser, id: int, state_in: StateUpdate
) -> Any:
    """
    Update an state.
    """
    state = session.get(State, id)
    if not state:
        raise HTTPException(status_code=404, detail="State not found")
    if not current_user.is_superuser and (state.owner_id != current_user.id):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    update_dict = state_in.model_dump(exclude_unset=True)
    state.sqlmodel_update(update_dict)
    session.add(state)
    session.commit()
    session.refresh(state)
    return state


@router.delete("/{id}")
def delete_state(session: SessionDep, current_user: CurrentUser, id: int) -> Message:
    """
    Delete an state.
    """
    state = session.get(State, id)
    if not state:
        raise HTTPException(status_code=404, detail=f"State {id} not found")
    if not current_user.is_superuser and (state.owner_id != current_user.id):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    session.delete(state)
    session.commit()
    return Message(message=f"State {id} deleted successfully")
