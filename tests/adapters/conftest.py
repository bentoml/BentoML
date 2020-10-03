# pylint: disable=redefined-outer-name

import pytest

from bentoml.adapters import DefaultOutput
from bentoml.service import InferenceAPI


@pytest.fixture(params=(True, False))
def batch_mode(request):
    return request.param


@pytest.fixture()
def make_api(batch_mode):
    def _make_api(input_adapter, user_func):
        if not input_adapter.BATCH_MODE_SUPPORTED and batch_mode:
            pytest.skip()
        if not input_adapter.SINGLE_MODE_SUPPORTED and not batch_mode:
            pytest.skip()
        return InferenceAPI(
            None,
            "test_api",
            "",
            input_adapter=input_adapter,
            user_func=user_func,
            output_adapter=DefaultOutput(),
            batch=batch_mode,
        )

    return _make_api
