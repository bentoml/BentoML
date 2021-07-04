import uuid

from click.testing import CliRunner

from bentoml.cli import create_bentoml_cli


def test_label_selectors_on_cli_list(bento_service):
    runner = CliRunner()
    cli = create_bentoml_cli()
    failed_result = runner.invoke(
        cli.commands['list'], ['--labels', '"incorrect label query"']
    )
    assert failed_result.exit_code == 1
    assert type(failed_result.exception) == AssertionError
    assert str(failed_result.exception) == 'Operator "label" is invalid'

    unique_label_value = uuid.uuid4().hex
    bento_service.save(
        labels={
            'test_id': unique_label_value,
            'example_key': 'values_can_contains_and-and.s',
        }
    )

    success_result = runner.invoke(
        cli.commands['list'], ['--labels', f'test_id in ({unique_label_value})']
    )
    assert success_result.exit_code == 0
    assert f'{bento_service.name}:{bento_service.version}' in success_result.output


def test_label_selectors_on_cli_get(bento_service):
    runner = CliRunner()
    cli = create_bentoml_cli()

    unique_label_value = uuid.uuid4().hex
    bento_service.save(labels={'test_id': unique_label_value})

    success_result = runner.invoke(
        cli.commands['get'],
        [bento_service.name, '--labels', f'test_id in ({unique_label_value})'],
    )
    assert success_result.exit_code == 0
    assert f'{bento_service.name}:{bento_service.version}' in success_result.output
