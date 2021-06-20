# pylint: disable=redefined-outer-name
import logging
import os
import uuid

import pytest

import bentoml
from bentoml.saved_bundle.loader import load_from_dir
from bentoml.utils.tempdir import TempDirectory
from bentoml.yatai.client import get_yatai_client
from tests.yatai.local_yatai_service import yatai_service_container

logger = logging.getLogger('bentoml.test')


@pytest.fixture(scope='session')
def yatai_service_url():
    with yatai_service_container() as yatai_url:
        yield yatai_url


class TestModel(object):
    def predict(self, input_data):
        return int(input_data) * 2


def test_save_load(yatai_service_url, example_bento_service_class):
    yc = get_yatai_client(yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)

    saved_path = svc.save(yatai_url=yatai_service_url)
    assert saved_path

    bento_pb = yc.repository.get(f'{svc.name}:{svc.version}')
    with TempDirectory() as temp_dir:
        new_temp_dir = os.path.join(temp_dir, uuid.uuid4().hex[:12])
        yc.repository.download_to_directory(bento_pb, new_temp_dir)
        bento_service = load_from_dir(new_temp_dir)
        assert bento_service.predict(1) == 2


def test_push(yatai_service_url, example_bento_service_class):
    yc = get_yatai_client(yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = svc.save()

    pushed_path = yc.repository.push(f'{svc.name}:{svc.version}')
    assert pushed_path != saved_path


def test_pull(yatai_service_url, example_bento_service_class):
    yc = get_yatai_client(yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = svc.save(yatai_url=yatai_service_url)

    pulled_local_path = yc.repository.pull(f'{svc.name}:{svc.version}')
    assert pulled_local_path != saved_path


def test_push_with_labels(yatai_service_url, example_bento_service_class):
    yc = get_yatai_client(yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = svc.save(labels={'foo': 'bar', 'abc': '123'})

    pushed_path = yc.repository.push(f'{svc.name}:{svc.version}')
    assert pushed_path != saved_path
    remote_bento_pb = yc.repository.get(f'{svc.name}:{svc.version}')
    assert remote_bento_pb.bento_service_metadata.labels
    labels = dict(remote_bento_pb.bento_service_metadata.labels)
    assert labels['foo'] == 'bar'
    assert labels['abc'] == '123'


def test_pull_with_labels(yatai_service_url, example_bento_service_class):
    yc = get_yatai_client(yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = svc.save(
        yatai_url=yatai_service_url, labels={'foo': 'bar', 'abc': '123'}
    )

    pulled_local_path = yc.repository.pull(f'{svc.name}:{svc.version}')
    assert pulled_local_path != saved_path
    local_yc = get_yatai_client()
    local_bento_pb = local_yc.repository.get(f'{svc.name}:{svc.version}')
    assert local_bento_pb.bento_service_metadata.labels
    labels = dict(local_bento_pb.bento_service_metadata.labels)
    assert labels['foo'] == 'bar'
    assert labels['abc'] == '123'


def test_get(yatai_service_url, example_bento_service_class):
    yc = get_yatai_client(yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    svc.save(yatai_url=yatai_service_url)
    svc_pb = yc.repository.get(f'{svc.name}:{svc.version}')
    assert svc_pb.bento_service_metadata.name == svc.name
    assert svc_pb.bento_service_metadata.version == svc.version


def test_list_by_name(yatai_service_url, example_bento_service_class):
    class ListByNameSvc(bentoml.BentoService):
        pass

    yc = get_yatai_client(yatai_service_url)
    svc = ListByNameSvc()
    svc.save(yatai_url=yatai_service_url)

    bentos = yc.repository.list(bento_name=svc.name)
    assert len(bentos) == 1


def test_load(yatai_service_url, example_bento_service_class):
    yc = get_yatai_client(yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    svc.save(yatai_url=yatai_service_url)

    loaded_svc = yc.repository.load(f'{svc.name}:{svc.version}')
    assert loaded_svc.name == svc.name


def test_load_from_dir(example_bento_service_class):
    yc = get_yatai_client()
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = svc.save()

    loaded_svc = yc.repository.load(saved_path)
    assert loaded_svc.name == svc.name
