# pylint: disable=redefined-outer-name

from typing import Callable, Optional
from unittest.mock import Mock

import pytest

from bentoml.adapters import DefaultOutput
from bentoml.adapters.base_input import BaseInputAdapter
from bentoml.adapters.base_output import BaseOutputAdapter
from bentoml.service import InferenceAPI


@pytest.fixture(params=(True, False))
def batch_mode(request):
    return request.param


@pytest.fixture()
def make_api(
    batch_mode,
) -> Callable[[BaseInputAdapter, Callable, Optional[BaseOutputAdapter]], InferenceAPI]:
    service_mock = Mock()
    service_mock.name = "TestBentoService"

    def _make_api(input_adapter, user_func, output_adapter=None, **kwargs):
        if not input_adapter.BATCH_MODE_SUPPORTED and batch_mode:
            pytest.skip()
        if not input_adapter.SINGLE_MODE_SUPPORTED and not batch_mode:
            pytest.skip()
        if output_adapter is None:
            output_adapter = DefaultOutput()
        return InferenceAPI(
            service_mock,
            "test_api",
            "",
            input_adapter=input_adapter,
            user_func=user_func,
            output_adapter=output_adapter,
            batch=batch_mode,
            **kwargs,
        )

    return _make_api
