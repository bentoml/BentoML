import pytest
import json

from bentoml.testing.utils import async_request
from tests.e2e.bento_server_general_features.service import svc


def test_routes_api():
    assert all(i.path.startswith("/") for i in svc.asgi_app.routes)


@pytest.mark.asyncio
async def test_api_routes(host: str) -> None:
    for obj in [1, 2.2, "str", [1, 2, 3], {"a": 1, "b": 2}]:
        obj_str = json.dumps(obj, separators=(",", ":"))
        await async_request(
            "POST",
            f"http://{host}/api/v1/test_route",
            headers=(("Content-Type", "application/json"),),
            data=obj_str,
            assert_status=200,
            assert_data=obj_str.encode("utf-8"),
        )
        await async_request(
            "POST",
            f"http://{host}/api/v1/with_prefix",
            headers=(("Content-Type", "application/json"),),
            data=obj_str,
            assert_status=200,
            assert_data=obj_str.encode("utf-8"),
        )
        await async_request(
            "POST",
            f"http://{host}//api/v1/with_prefix",
            assert_status=404,
        )
