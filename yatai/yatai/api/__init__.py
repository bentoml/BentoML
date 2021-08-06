from fastapi import APIRouter

from yatai.api.api_v1.bundles import router as bundle_router

api_router = APIRouter()

api_router.include_router(bundle_router, prefix="api/v1", tags=["bundles"])
