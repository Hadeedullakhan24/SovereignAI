from fastapi import APIRouter, Depends
from backend.app.api.dependencies import current_user
from backend.app.models import User
router=APIRouter(prefix="/agents",tags=["agents"])
@router.get("")
def list_agents(user:User=Depends(current_user)): return [{"id":"sovereign","name":"Sovereign Agent","providers":["local"]}]
