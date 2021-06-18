import pytest

from bentoml.exceptions import InvalidArgument
from bentoml.yatai.db.stores.label import _validate_labels


def test_validate_labels_fails():
    with pytest.raises(InvalidArgument):
        _validate_labels(
            {'this_is_a_super_long_key_name_it_will_be_more_than_the_max_allowed': 'v'}
        )
    with pytest.raises(InvalidArgument):
        _validate_labels({'key_contains!': 'value'})
    with pytest.raises(InvalidArgument):
        _validate_labels({'key': 'value-contains?'})
    with pytest.raises(InvalidArgument):
        _validate_labels({'key nop': 'value'})
    with pytest.raises(InvalidArgument):
        _validate_labels({'key': '1', 'key3@#!$': 'value'})
    with pytest.raises(InvalidArgument):
        _validate_labels({'key': 'cant_end_with_symbol_'})


def test_validate_labels_pass():
    _validate_labels({'long_key_title': 'some_value', 'another_key': "value"})
    _validate_labels({'long_key-title': 'some_value-inside.this'})
    _validate_labels({'create_by': 'admin', 'py.version': '3.6.8'})
