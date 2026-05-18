from fastapi import APIRouter
from app.api.v1.endpoints import load, query

api_router = APIRouter()
api_router.include_router(load.router, prefix="/load", tags=["load"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
