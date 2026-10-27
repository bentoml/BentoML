from __future__ import annotations

import sys
import typing as t

import pytest
from pydantic import BaseModel
from starlette.datastructures import FormData

from _bentoml_impl.serde import JSONSerde
from _bentoml_impl.serde import MultipartSerde
from _bentoml_impl.serde import Payload
from _bentoml_impl.serde import _is_str_annotation


class _FakeRequest:
    """Minimal stand-in for a starlette request carrying a parsed form."""

    def __init__(self, form: FormData) -> None:
        self._form = form

    async def form(self) -> FormData:
        return self._form


class _EchoInput(BaseModel):
    text: str = ""
    opt: t.Optional[str] = None
    n: int = 0
    tags: t.List[str] = []

    multipart_fields: t.ClassVar[set[str]] = set()


# NOTE: `str | None` (PEP 604) cannot be evaluated on Python 3.9 at all, so
# it must not appear in a module-level parametrize list: a skipif mark would
# not help, since the expression is still evaluated at decoration time.
# Build the case list conditionally instead; on 3.9 the expression below is
# never executed, so test collection succeeds there.
_IS_STR_ANNOTATION_CASES: list[tuple[t.Any, bool]] = [
    (str, True),
    (t.Optional[str], True),
    (int, False),
    (t.List[str], False),
    (t.Union[str, int], False),
]
if sys.version_info >= (3, 10):
    _IS_STR_ANNOTATION_CASES.append((str | None, True))


@pytest.mark.parametrize("annotation,expected", _IS_STR_ANNOTATION_CASES)
def test_is_str_annotation(annotation: t.Any, expected: bool) -> None:
    assert _is_str_annotation(annotation) is expected


@pytest.mark.parametrize(
    "value",
    ["hello", "00123", "123", "1.5", "true", "null", "[1]", '"quoted"'],
)
@pytest.mark.asyncio
async def test_multipart_str_field_keeps_raw_string(value: str) -> None:
    # Regression test for bentoml/BentoML#5766: values that happen to be
    # valid JSON must reach a str field verbatim instead of being parsed
    # into int/float/bool/None/list (HTTP 400) or unquoted.
    result = await MultipartSerde().parse_request(
        _FakeRequest(FormData([("text", value)])), _EchoInput
    )
    assert result.text == value
    assert isinstance(result.text, str)


@pytest.mark.asyncio
async def test_multipart_optional_str_field_keeps_raw_string() -> None:
    result = await MultipartSerde().parse_request(
        _FakeRequest(FormData([("opt", "123")])), _EchoInput
    )
    assert result.opt == "123"
    assert isinstance(result.opt, str)


@pytest.mark.asyncio
async def test_multipart_non_str_fields_still_parse_json() -> None:
    result = await MultipartSerde().parse_request(
        _FakeRequest(FormData([("n", "5"), ("tags", '["a", "b"]')])), _EchoInput
    )
    assert result.n == 5
    assert result.tags == ["a", "b"]


def test_json_path_unchanged() -> None:
    result = JSONSerde().deserialize_model(Payload((b'{"text": "123"}',)), _EchoInput)
    assert result.text == "123"
