# pylint: disable=redefined-outer-name
import uuid
import pytest

from click.testing import CliRunner

from bentoml import BentoService
from bentoml.cli import create_bentoml_cli
from bentoml.yatai.client import get_yatai_client
from bentoml.exceptions import BentoMLException


@pytest.fixture()
def yatai_client():
    yc = get_yatai_client()
    yc.repository.delete(prune=True)
    assert len(yc.repository.list()) == 0
    yield yc
    yc.repository.delete(prune=True)


def test_delete_bento_by_tag(bento_service, yatai_client):
    bento_service.save()
    bento_tag = f"{bento_service.name}:{bento_service.version}"

    yatai_client.repository.get(bento_tag)
    yatai_client.repository.delete(
        bento_name=bento_service.name, bento_version=bento_service.version,
    )
    with pytest.raises(BentoMLException) as excinfo:
        yatai_client.repository.get(bento_tag)
    assert f"{bento_tag} not found" in str(excinfo.value)


def test_delete_bentos_by_labels(bento_service, yatai_client):
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={'cohort': '100'})
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={'cohort': '110'})
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={'cohort': '120'})

    bentos = yatai_client.repository.list(labels='cohort')
    assert len(bentos) == 3

    yatai_client.repository.delete(labels='cohort in (100, 110)')
    bentos = yatai_client.repository.list(labels='cohort')
    assert len(bentos) == 1

    yatai_client.repository.delete(labels='cohort')
    bentos = yatai_client.repository.list(labels='cohort')
    assert len(bentos) == 0


def test_delete_bento_by_name(yatai_client):
    class DeleteBentoByNameTest(BentoService):
        pass

    svc = DeleteBentoByNameTest()
    svc.save()

    assert yatai_client.repository.get(svc.tag).version == svc.version
    assert len(yatai_client.repository.list(bento_name=svc.name)) == 1

    yatai_client.repository.delete(bento_name=svc.name)
    with pytest.raises(BentoMLException) as excinfo:
        yatai_client.repository.get(svc.tag)
    assert f"{svc.tag} not found" in str(excinfo.value)
    assert len(yatai_client.repository.list(bento_name=svc.name)) == 0


def test_delete_bentos_by_name_and_labels(bento_service, yatai_client):
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={'dataset': '20201212'})
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={'dataset': '20201212'})
    bento_service.save(version=uuid.uuid4().hex[0:8], labels={'dataset': '20210101'})

    class ThisShouldNotBeDeleted(BentoService):
        pass

    svc2 = ThisShouldNotBeDeleted()
    svc2.save()

    yatai_client.repository.delete(
        bento_name=bento_service.name, labels='dataset=20201212'
    )
    assert len(yatai_client.repository.list(bento_name=bento_service.name)) == 1

    # other services should not be deleted
    assert yatai_client.repository.get(svc2.tag).version == svc2.version
    yatai_client.repository.delete(svc2.tag)
    assert len(yatai_client.repository.list(bento_name=svc2.name)) == 0


def test_delete_multiple_bentos_by_tag_from_cli(bento_service, yatai_client):
    version_one = uuid.uuid4().hex[0:8]
    version_two = uuid.uuid4().hex[0:8]
    version_three = uuid.uuid4().hex[0:8]
    bento_service.save(version=version_one)
    bento_service.save(version=version_two)
    bento_service.save(version=version_three)

    runner = CliRunner()
    cli = create_bentoml_cli()
    result = runner.invoke(
        cli.commands['delete'], [f'{bento_service.name}:{version_one}', '-y'],
    )
    assert result.exit_code == 0
    assert "Deleted" in result.output
    assert len(yatai_client.repository.list(bento_name=bento_service.name)) == 2

    result = runner.invoke(
        cli.commands['delete'],
        [
            f'{bento_service.name}:{version_two},{bento_service.name}:{version_three}',
            '-y',
        ],
    )
    assert result.exit_code == 0
    assert len(yatai_client.repository.list(bento_name=bento_service.name)) == 0


def test_delete_all_bentos(bento_service, yatai_client):
    bento_service.save(version=uuid.uuid4().hex[0:8])
    bento_service.save(version=uuid.uuid4().hex[0:8])
    bento_service.save(version=uuid.uuid4().hex[0:8])
    assert len(yatai_client.repository.list()) == 3

    yatai_client.repository.delete(prune=True)
    assert len(yatai_client.repository.list()) == 0


def test_delete_all_bentos_from_cli(bento_service, yatai_client):
    bento_service.save(version=uuid.uuid4().hex[0:8])
    bento_service.save(version=uuid.uuid4().hex[0:8])
    bento_service.save(version=uuid.uuid4().hex[0:8])
    assert len(yatai_client.repository.list()) == 3

    runner = CliRunner()
    cli = create_bentoml_cli()
    result = runner.invoke(cli.commands['delete'], ['--all', '-y'],)
    assert result.exit_code == 0
    assert len(yatai_client.repository.list()) == 0
