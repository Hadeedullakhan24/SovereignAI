from fastapi import APIRouter
from .endpoints import auth,chat,files,documents,tasks,results,health
router=APIRouter(prefix="/api/v1")
for module in (auth,chat,files,documents,tasks,results,health): router.include_router(module.router)
