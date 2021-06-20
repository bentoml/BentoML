import logging
import uuid

from click.testing import CliRunner

from bentoml.cli import create_bentoml_cli
from tests.bento_service_examples.iris_classifier import IrisClassifier
from bentoml.yatai.client import get_yatai_client

logger = logging.getLogger("bentoml.test")


def test_delete_single_bento(bento_service):
    yc = get_yatai_client()
    # Remove all other bentos. Clean state
    deleted_version = uuid.uuid4().hex[0:8]
    yc.repository.delete(prune=True)

    bento_service.save(version=deleted_version)
    yc.repository.delete(
        bento_name=bento_service.name,
        bento_version=deleted_version,
    )
    bentos = yc.repository.list()
    assert len(bentos) == 0

    another_deleted_version = uuid.uuid4().hex[0:10]
    bento_service.save(version=another_deleted_version)
    yc.repository.delete(f"{bento_service.name}:{bento_service.version}")
    bentos = yc.repository.list()
    assert len(bentos) == 0

    # Clean up existing bentos
    yc.repository.delete(prune=True)


def test_delete_bentos_base_on_labels(bento_service):
    yc = get_yatai_client()
    yc.repository.delete(prune=True)
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={"cohort": "100"})
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={"cohort": "110"})
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={"cohort": "120"})

    yc.repository.delete(labels="cohort in (100, 110)")
    bentos = yc.repository.list()
    assert len(bentos) == 1
    # Clean up existing bentos
    yc.repository.delete(prune=True)


def test_delete_bentos_base_on_name(bento_service):
    yc = get_yatai_client()
    yc.repository.delete(prune=True)
    bento_service.save(version=uuid.uuid4().hex[0:8])
    bento_service.save(version=uuid.uuid4().hex[0:8])
    iris = IrisClassifier()
    iris.save()

    yc.repository.delete(bento_name=bento_service.name)
    bentos = yc.repository.list()
    assert len(bentos) == 1
    # Clean up existing bentos
    yc.repository.delete(prune=True)


def test_delete_bentos_on_name_and_labels(bento_service):
    yc = get_yatai_client()
    yc.repository.delete(prune=True)
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={"dataset": "20201212"})
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={"dataset": "20201212"})
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={"dataset": "20210101"})
    iris = IrisClassifier()
    iris.save()

    yc.repository.delete(bento_name=bento_service.name, labels="dataset=20201212")
    bentos = yc.repository.list()
    assert len(bentos) == 2
    # Clean up existing bentos
    yc.repository.delete(prune=True)


def test_delete_bentos_with_tags(bento_service):
    yc = get_yatai_client()
    yc.repository.delete(prune=True)
    version_one = uuid.uuid4().hex[0:8]
    version_two = uuid.uuid4().hex[0:8]
    version_three = uuid.uuid4().hex[0:8]
    bento_service.save(version=version_one)
    bento_service.save(version=version_two)
    bento_service.save(version=version_three)
    yc.repository.delete(bento_tag=f"{bento_service.name}:{version_one}")
    bentos = yc.repository.list()
    assert len(bentos) == 2

    runner = CliRunner()
    cli = create_bentoml_cli()
    print(cli.commands["delete"])
    runner.invoke(
        cli.commands["delete"],
        [
            f"{bento_service.name}:{version_two},{bento_service.name}:{version_three}",
            "-y",
        ],
    )
    bentos = yc.repository.list()
    assert len(bentos) == 0
    # Clean up existing bentos
    yc.repository.delete(prune=True)


def test_delete_all_bentos(bento_service):
    bento_service.save(version=uuid.uuid4().hex[0:8])
    bento_service.save(version=uuid.uuid4().hex[0:8])
    bento_service.save(version=uuid.uuid4().hex[0:8])

    yc = get_yatai_client()
    yc.repository.delete(prune=True)
    bentos = yc.repository.list()
    assert len(bentos) == 0

    bento_service.save(version=uuid.uuid4().hex[0:8])
    bento_service.save(version=uuid.uuid4().hex[0:8])
    bento_service.save(version=uuid.uuid4().hex[0:8])

    runner = CliRunner()
    cli = create_bentoml_cli()
    runner.invoke(
        cli.commands["delete"],
        ["--all", "-y"],
    )
    bentos = yc.repository.list()
    assert len(bentos) == 0
