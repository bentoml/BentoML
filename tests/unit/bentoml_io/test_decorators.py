import pytest

import bentoml


@pytest.mark.asyncio
async def test_mount_asgi_app():
    import httpx
    from fastapi import FastAPI

    app = FastAPI()

    @bentoml.mount_asgi_app(app, path="/test")
    @bentoml.service(metrics={"enabled": False})
    class TestService:
        @app.get("/hello")
        def hello(self):
            return {"message": "Hello, world!"}

    async with httpx.AsyncClient(
        app=TestService.to_asgi(init=True), base_url="http://testserver"
    ) as client:
        response = await client.get("/test/hello")
        assert response.status_code == 200
        assert response.json()["message"] == "Hello, world!"


@pytest.mark.asyncio
async def test_mount_asgi_app_later():
    import httpx
    from fastapi import FastAPI

    app = FastAPI()

    @bentoml.service(metrics={"enabled": False})
    @bentoml.mount_asgi_app(app, path="/test")
    class TestService:
        @app.get("/hello")
        def hello(self):
            return {"message": "Hello, world!"}

    async with httpx.AsyncClient(
        app=TestService.to_asgi(init=True), base_url="http://testserver"
    ) as client:
        response = await client.get("/test/hello")
        assert response.status_code == 200
        assert response.json()["message"] == "Hello, world!"
