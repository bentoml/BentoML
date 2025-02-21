from __future__ import annotations

from bentoml._internal.utils.merge import deep_merge


def test_deep_merge():
    # Test basic merge
    assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    # Test nested merge
    assert deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}

    # Test override
    assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    # Test nested override
    assert deep_merge({"a": {"b": 1}}, {"a": 2}) == {"a": 2}

    # Test empty dicts
    assert deep_merge({}, {"a": 1}) == {"a": 1}
    assert deep_merge({"a": 1}, {}) == {"a": 1}
    assert deep_merge({}, {}) == {}

    # Test complex nested structures
    assert deep_merge(
        {"a": {"b": {"c": 1}}, "d": 2}, {"a": {"b": {"d": 3}}, "e": 4}
    ) == {"a": {"b": {"c": 1, "d": 3}}, "d": 2, "e": 4}

    # Test list handling
    assert deep_merge({"a": [1, 2]}, {"a": [3, 4]}) == {
        "a": [3, 4]
    }  # Lists should be overridden like other types

    # Test None handling
    assert deep_merge({"a": None}, {"a": {"b": 1}}) == {"a": {"b": 1}}
    assert deep_merge({"a": {"b": 1}}, {"a": None}) == {"a": None}
