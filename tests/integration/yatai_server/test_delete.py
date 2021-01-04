import logging
import uuid

from tests.bento_service_examples.iris_classifier import IrisClassifier
from bentoml.yatai.client import get_yatai_client

logger = logging.getLogger('bentoml.test')


def test_delete_single_bento(bento_service):
    yc = get_yatai_client()
    # Remove all other bentos. Clean state
    deleted_version = uuid.uuid4().hex[0:8]
    yc.repository.delete(prune=True, confirm_delete=True)

    bento_service.save(version=deleted_version)
    bento_service.save(version=uuid.uuid4().hex[0:8])
    yc.repository.delete(
        bento_name=bento_service.name,
        bento_version=deleted_version,
        confirm_delete=True,
    )
    bentos = yc.repository.list()
    assert len(bentos) == 1
    # Clean up existing bentos
    yc.repository.delete(prune=True, confirm_delete=True)


def test_delete_bentos_base_on_labels(bento_service):
    yc = get_yatai_client()
    yc.repository.delete(prune=True, confirm_delete=True)
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={'cohort': '100'})
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={'cohort': '110'})
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={'cohort': '120'})

    yc.repository.delete(labels='cohort in (100, 110)', confirm_delete=True)
    bentos = yc.repository.list()
    assert len(bentos) == 1
    # Clean up existing bentos
    yc.repository.delete(prune=True, confirm_delete=True)


def test_delete_bentos_base_on_name(bento_service):
    yc = get_yatai_client()
    yc.repository.delete(prune=True, confirm_delete=True)
    bento_service.save(version=uuid.uuid4().hex[0:8])
    bento_service.save(version=uuid.uuid4().hex[0:8])
    iris = IrisClassifier()
    iris.save()

    yc.repository.delete(bento_name=bento_service.name, confirm_delete=True)
    bentos = yc.repository.list()
    assert len(bentos) == 1
    # Clean up existing bentos
    yc.repository.delete(prune=True, confirm_delete=True)


def test_delete_bentos_on_name_and_labels(bento_service):
    yc = get_yatai_client()
    yc.repository.delete(prune=True, confirm_delete=True)
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={'dataset': '20201212'})
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={'dataset': '20201212'})
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={'dataset': '20210101'})
    iris = IrisClassifier()
    iris.save()

    yc.repository.delete(
        bento_name=bento_service.name, labels='dataset=20201212', confirm_delete=True
    )
    bentos = yc.repository.list()
    assert len(bentos) == 2
    # Clean up existing bentos
    yc.repository.delete(prune=True, confirm_delete=True)


def test_delete_all_bentos(bento_service):
    bento_service.save(version=uuid.uuid4().hex[0:8])
    bento_service.save(version=uuid.uuid4().hex[0:8])
    bento_service.save(version=uuid.uuid4().hex[0:8])

    yc = get_yatai_client()
    yc.repository.delete(prune=True, confirm_delete=True)
    bentos = yc.repository.list()
    assert len(bentos) == 0
