import pytest

import bentoml


@pytest.mark.asyncio
async def test_mount_asgi_app():
    import httpx
    from fastapi import FastAPI

    from _bentoml_impl.server.app import ServiceAppFactory

    app = FastAPI()

    @bentoml.mount_asgi_app(app, path="/test")
    @bentoml.service
    class TestService:
        @app.get("/hello")
        def hello(self):
            return {"message": "Hello, world!"}

    TestService.inject_config()

    factory = ServiceAppFactory(TestService, is_main=True)
    factory.create_instance()
    try:
        async with httpx.AsyncClient(
            app=factory(), base_url="http://testserver"
        ) as client:
            response = await client.get("/test/hello")
            assert response.status_code == 200
            assert response.json()["message"] == "Hello, world!"
    finally:
        await factory.destroy_instance()


@pytest.mark.asyncio
async def test_mount_asgi_app_later():
    import httpx
    from fastapi import FastAPI

    from _bentoml_impl.server.app import ServiceAppFactory

    app = FastAPI()

    @bentoml.service
    @bentoml.mount_asgi_app(app, path="/test")
    class TestService:
        @app.get("/hello")
        def hello(self):
            return {"message": "Hello, world!"}

    TestService.inject_config()

    factory = ServiceAppFactory(TestService, is_main=True)
    factory.create_instance()
    try:
        async with httpx.AsyncClient(
            app=factory(), base_url="http://testserver"
        ) as client:
            response = await client.get("/test/hello")
            assert response.status_code == 200
            assert response.json()["message"] == "Hello, world!"
    finally:
        await factory.destroy_instance()
