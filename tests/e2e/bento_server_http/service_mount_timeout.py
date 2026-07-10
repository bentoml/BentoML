from __future__ import annotations

import asyncio

from fastapi import FastAPI

import bentoml

# Create a FastAPI app with a delayed endpoint
fastapi_app = FastAPI()


@fastapi_app.get("/delay")
async def delay_endpoint():
    # Delay for 2 seconds
    await asyncio.sleep(2)
    return {"message": "delayed response"}


# Create a BentoML service without any runners (since we are only mounting an ASGI app)
svc = bentoml.legacy.Service(name="mounted_timeout_service", runners=[])

# Mount the FastAPI app at the path "/mounted"
svc.mount_asgi_app(fastapi_app, path="/mounted")
