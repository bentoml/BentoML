import click
import pytest

from bentoml.cli.bento_management import parse_delete_targets_argument_callback


def test_parse_delete_targets_argument_callback():
    list_of_valid_values = 'ABC,test:123'
    list_of_values_with_invalid_value = 'test:123,,!@#'
    valid_value = 'Afs:1232fdsa'
    invalid_value = 'dfds$%@'

    valid_list_result = parse_delete_targets_argument_callback(
        ctx=None, params=None, value=list_of_valid_values
    )
    assert len(valid_list_result) == 2
    valid_value_result = parse_delete_targets_argument_callback(
        ctx=None, params=None, value=valid_value
    )
    assert len(valid_value_result) == 1

    with pytest.raises(click.BadParameter):
        parse_delete_targets_argument_callback(
            ctx=None, params=None, value=list_of_values_with_invalid_value
        )

    with pytest.raises(click.BadParameter):
        parse_delete_targets_argument_callback(
            ctx=None, params=None, value=invalid_value
        )
