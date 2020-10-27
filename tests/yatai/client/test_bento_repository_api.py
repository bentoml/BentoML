import uuid
import logging

import pytest

from bentoml.exceptions import InvalidArgument
from bentoml.yatai.client import YataiClient
from bentoml.yatai.label_store import _validate_labels

logger = logging.getLogger('bentoml.test')


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


def test_dangerously_delete(example_bento_service_class):
    yc = YataiClient()
    svc = example_bento_service_class()
    version = f"test_{uuid.uuid4().hex}"
    svc.set_version(version_str=version)
    svc.save()
    result = yc.repository.dangerously_delete_bento(svc.name, svc.version)
    assert result.status.status_code == 0
    logger.debug('Saving the bento service again with the same name:version')
    svc.save()
    second_delete_result = yc.repository.dangerously_delete_bento(svc.name, svc.version)
    assert second_delete_result.status.status_code == 0
