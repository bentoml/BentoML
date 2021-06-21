# pylint: disable=redefined-outer-name
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

    yatai_client.repository.get(bento_service.tag)
    yatai_client.repository.delete(
        bento_name=bento_service.name, bento_version=bento_service.version,
    )
    with pytest.raises(BentoMLException) as excinfo:
        yatai_client.repository.get(bento_service.tag)
    assert f"{bento_service.tag} not found" in str(excinfo.value)


def test_delete_bentos_by_labels(example_bento_service_class, yatai_client):
    svc1 = example_bento_service_class()
    svc1.save(labels={'cohort': '100'})
    svc2 = example_bento_service_class()
    svc2.save(labels={'cohort': '110'})
    svc3 = example_bento_service_class()
    svc3.save(labels={'cohort': '120'})

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


def test_delete_bentos_by_name_and_labels(example_bento_service_class, yatai_client):
    svc1 = example_bento_service_class()
    svc1.save(labels={'dataset': '20201212'})
    svc2 = example_bento_service_class()
    svc2.save(labels={'dataset': '20201212'})
    svc3 = example_bento_service_class()
    svc3.save(labels={'dataset': '20210101'})

    class ThisShouldNotBeDeleted(BentoService):
        pass

    svc2 = ThisShouldNotBeDeleted()
    svc2.save()

    yatai_client.repository.delete(
        bento_name=example_bento_service_class.name, labels='dataset=20201212'
    )
    assert (
        len(yatai_client.repository.list(bento_name=example_bento_service_class.name))
        == 1
    )

    # other services should not be deleted
    assert yatai_client.repository.get(svc2.tag).version == svc2.version
    yatai_client.repository.delete(svc2.tag)
    assert len(yatai_client.repository.list(bento_name=svc2.name)) == 0


def test_delete_all_bentos(example_bento_service_class, yatai_client):
    svc1 = example_bento_service_class()
    svc1.save()
    svc2 = example_bento_service_class()
    svc2.save()
    svc3 = example_bento_service_class()
    svc3.save()

    assert len(yatai_client.repository.list()) == 3

    yatai_client.repository.delete(prune=True)
    assert len(yatai_client.repository.list()) == 0


def test_delete_all_bentos_from_cli(example_bento_service_class, yatai_client):
    svc1 = example_bento_service_class()
    svc1.save()
    svc2 = example_bento_service_class()
    svc2.save()
    svc3 = example_bento_service_class()
    svc3.save()

    assert len(yatai_client.repository.list()) == 3

    runner = CliRunner()
    cli = create_bentoml_cli()
    result = runner.invoke(cli.commands['delete'], ['--all', '-y'],)
    assert result.exit_code == 0
    assert len(yatai_client.repository.list()) == 0


def test_delete_multiple_bentos_by_tag_from_cli(
    example_bento_service_class, yatai_client
):
    svc1 = example_bento_service_class()
    svc1.save()
    svc2 = example_bento_service_class()
    svc2.save()
    svc3 = example_bento_service_class()
    svc3.save()

    runner = CliRunner()
    cli = create_bentoml_cli()
    result = runner.invoke(cli.commands['delete'], [svc1.tag, '-y'],)
    assert result.exit_code == 0
    assert (
        len(yatai_client.repository.list(bento_name=example_bento_service_class.name))
        == 2
    )

    result = runner.invoke(cli.commands['delete'], [f'{svc2.tag},{svc3.tag}', '-y'])
    assert result.exit_code == 0
    assert (
        len(yatai_client.repository.list(bento_name=example_bento_service_class.name))
        == 0
    )
