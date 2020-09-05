import pytest

from bentoml.service import InferenceAPI
from bentoml.adapters import DefaultOutput


@pytest.fixture()
def make_api():
    def _make_api(input_adapter, user_func):
        return InferenceAPI(
            None,
            "test_api",
            "",
            input_adapter=input_adapter,
            user_func=user_func,
            output_adapter=DefaultOutput(),
        )

    return _make_api
