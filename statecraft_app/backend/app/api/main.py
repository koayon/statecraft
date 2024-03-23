from app.api.routes import login, users, utils
from fastapi import APIRouter

from statecraft_app.backend.app.api.routes import states

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(utils.router, prefix="/utils", tags=["utils"])
api_router.include_router(states.router, prefix="/states", tags=["states"])
