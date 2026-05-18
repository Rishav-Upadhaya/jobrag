import sys

from app.api.v1.endpoints import load as load_module
from app.api.v1.endpoints import query as query_module

load = load_module
query = query_module
load_router = load_module.router
query_router = query_module.router

sys.modules.setdefault("app.api.load", load_module)
sys.modules.setdefault("app.api.query", query_module)

__all__ = ["load", "query", "load_router", "query_router"]
