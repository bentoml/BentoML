from __future__ import annotations

import typing as t

import pytest

from bentoml._internal.utils.unflatten import unflatten


@pytest.mark.parametrize(
    ("flat", "expected"),
    [
        # Empty input.
        ({}, {}),
        # Plain terminal values.
        ({"foo": 1}, {"foo": 1}),
        ({"foo": "bar", "baz": 2}, {"foo": "bar", "baz": 2}),
        # Nested dicts.
        ({"foo.bar": "val"}, {"foo": {"bar": "val"}}),
        ({"a.b.c.d": 1}, {"a": {"b": {"c": {"d": 1}}}}),
        ({"a.b": 1, "a.c": 2}, {"a": {"b": 1, "c": 2}}),
        # Nested lists.
        ({"foo[0]": "val", "foo[1]": "bar"}, {"foo": ["val", "bar"]}),
        ({"foo[0][0]": "val"}, {"foo": [["val"]]}),
        # Lists of dicts.
        (
            {"foo[0].bar": "val", "foo[1].baz": "x"},
            {"foo": [{"bar": "val"}, {"baz": "x"}]},
        ),
        # Empty-string terminal value is preserved (not treated as missing).
        ({"foo.bar": ""}, {"foo": {"bar": ""}}),
    ],
)
def test_unflatten(flat: dict[str, t.Any], expected: dict[str, t.Any]) -> None:
    assert unflatten(flat) == expected


def test_unflatten_accepts_sequence_of_pairs() -> None:
    # Any iterable of (key, value) pairs is accepted, not only dicts.
    assert unflatten([("foo.bar", "v"), ("foo.baz", "w")]) == {
        "foo": {"bar": "v", "baz": "w"}
    }


def test_unflatten_quoted_key_keeps_literal_dot() -> None:
    # A quoted segment is treated as a single literal key, dots included.
    assert unflatten({'"foo.bar"': "v"}) == {"foo.bar": "v"}


def test_unflatten_conflicting_dict_and_list() -> None:
    with pytest.raises(ValueError, match=r"conflicting types .* for key 'foo'"):
        unflatten({"foo.bar": 1, "foo[0]": 2})


def test_unflatten_conflicting_terminal_then_holder() -> None:
    with pytest.raises(ValueError, match="conflicting types terminal"):
        unflatten({"foo": 1, "foo.bar": 2})


def test_unflatten_conflicting_holder_then_terminal() -> None:
    with pytest.raises(ValueError, match="and terminal for key 'foo'"):
        unflatten({"foo.bar": 2, "foo": 1})


def test_unflatten_missing_list_index() -> None:
    with pytest.raises(ValueError, match=r"missing key 'foo\[1\]'"):
        unflatten({"foo[0]": "a", "foo[2]": "c"})


def test_unflatten_non_string_key() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        unflatten({1: "v"})
