from __future__ import annotations

import pytest

from bentoml import Tag
from bentoml.exceptions import BentoMLException


@pytest.mark.parametrize("value", ["a\n", "my_svc\n", "svc\n\n"])
def test_tag_name_rejects_trailing_newline(value: str) -> None:
    # ``$`` matches before a final newline, so a newline-terminated name used to pass.
    with pytest.raises(ValueError, match="not a valid BentoML tag"):
        Tag(value)


def test_tag_version_rejects_trailing_newline() -> None:
    with pytest.raises(ValueError, match="not a valid BentoML tag"):
        Tag("svc", "v1\n")
    with pytest.raises(BentoMLException, match="Invalid Tag"):
        Tag.from_str("svc:v1\n")


def test_tag_still_accepts_valid_values() -> None:
    assert str(Tag("my_svc", "v1.2-rc_3")) == "my_svc:v1.2-rc_3"
    assert Tag.from_str("svc:v1").version == "v1"
