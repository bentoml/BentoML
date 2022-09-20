from __future__ import annotations

import typing as t
from http import HTTPStatus
from unittest.mock import Mock

import grpc
import pytest

from bentoml.exceptions import BadInput
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import BentoMLException
from bentoml.grpc.utils import MethodName
from bentoml.grpc.utils import to_http_status
from bentoml.grpc.utils import grpc_status_code
from bentoml.grpc.utils import wrap_rpc_handler
from bentoml.grpc.utils import parse_method_name


@pytest.mark.parametrize(
    "exception,expected",
    [
        (BentoMLException, grpc.StatusCode.INTERNAL),
        (InvalidArgument, grpc.StatusCode.INVALID_ARGUMENT),
        (BadInput, grpc.StatusCode.INVALID_ARGUMENT),
        (
            type(
                "UnknownException",
                (BentoMLException,),
                {"error_code": HTTPStatus.ALREADY_REPORTED},
            ),
            grpc.StatusCode.UNKNOWN,
        ),
    ],
)
def test_exception_to_grpc_status(
    exception: t.Type[BentoMLException], expected: grpc.StatusCode
):
    assert grpc_status_code(exception("something")) == expected


@pytest.mark.parametrize(
    "status_code,expected",
    [
        (grpc.StatusCode.OK, HTTPStatus.OK),
        (grpc.StatusCode.CANCELLED, HTTPStatus.INTERNAL_SERVER_ERROR),
        (grpc.StatusCode.INVALID_ARGUMENT, HTTPStatus.BAD_REQUEST),
    ],
)
def test_grpc_to_http_status_code(status_code: grpc.StatusCode, expected: HTTPStatus):
    assert to_http_status(status_code) == expected


def test_method_name():
    # Fields are correct and fully_qualified_service work.
    mn = MethodName("foo.bar", "SearchService", "Search")
    assert mn.package == "foo.bar"
    assert mn.service == "SearchService"
    assert mn.method == "Search"
    assert mn.fully_qualified_service == "foo.bar.SearchService"


def test_empty_package_method_name():
    # fully_qualified_service works when there's no package
    mn = MethodName("", "SearchService", "Search")
    assert mn.fully_qualified_service == "SearchService"


def test_parse_method_name():
    mn, ok = parse_method_name("/foo.bar.SearchService/Search")
    assert mn.package == "foo.bar"
    assert mn.service == "SearchService"
    assert mn.method == "Search"
    assert ok


def test_parse_empty_package():
    # parse_method_name works with no package.
    mn, _ = parse_method_name("/SearchService/Search")
    assert mn.package == ""
    assert mn.service == "SearchService"
    assert mn.method == "Search"


@pytest.mark.parametrize(
    "request_streaming,response_streaming,handler_fn",
    [
        (True, True, "stream_stream"),
        (True, False, "stream_unary"),
        (False, True, "unary_stream"),
        (False, False, "unary_unary"),
    ],
)
def test_wrap_rpc_handler(
    request_streaming: bool,
    response_streaming: bool,
    handler_fn: str,
):
    mock_handler = Mock(
        request_streaming=request_streaming,
        response_streaming=response_streaming,
    )
    fn = Mock()
    assert wrap_rpc_handler(fn, None) is None
    # wrap_rpc_handler works with None handler.
    wrapped = wrap_rpc_handler(fn, mock_handler)
    assert fn.call_count == 1
    assert getattr(wrapped, handler_fn) is not None
