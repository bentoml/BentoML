from __future__ import annotations

import pytest

from bentoml._internal.utils import filter_control_codes


@pytest.mark.parametrize(
    "code",
    [
        "\x1b[H",  # home
        "\x1b[2J",  # clear
        "\x1b[?1049h",  # enable the alternate screen
        "\x1b[?1049l",  # disable the alternate screen
        "\x1b[?25h",  # show the cursor
        "\x1b[?25l",  # hide the cursor
        "\x1b[3A",  # cursor up
        "\x1b]0;title\x07",  # set the window title
    ],
)
def test_filter_control_codes_strips_terminal_codes(code: str) -> None:
    assert filter_control_codes(f"a{code}b") == "ab"


def test_filter_control_codes_keeps_plain_text() -> None:
    assert filter_control_codes("hello [?25h world") == "hello [?25h world"
