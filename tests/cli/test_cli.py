import os
import tempfile

import pytest
import mock
import click
from humanfriendly import format_size
from click.testing import CliRunner
import psutil  # noqa # pylint: disable=unused-import

from bentoml.cli.bento_service import (
    create_bento_service_cli,
    to_valid_docker_image_name,
    to_valid_docker_image_version,
    validate_tag,
)
from bentoml.cli.utils import echo_docker_api_result
from bentoml.exceptions import BentoMLException


def generate_test_input_file():
    import uuid

    random_id = uuid.uuid4().hex
    tempdir = tempfile.mkdtemp()
    file_path = os.path.join(tempdir, random_id + ".json")

    with open(file_path, "w") as f:
        f.write('[{"col1": 1}, {"col1": 2}]')
    return file_path


def assert_equal_lists(res, expected):
    assert len(expected) == len(res)
    assert all([a == b for a, b in zip(expected, res)])


@pytest.mark.parametrize(
    "name, expected_name",
    [
        ("ALLCAPS", "allcaps"),
        ("...as.df...", "as.df"),
        ("_asdf_asdf", "asdf_asdf"),
        ("1234-asdf--", "1234-asdf"),
    ],
)
def test_to_valid_docker_image_name(name, expected_name):
    assert to_valid_docker_image_name(name) == expected_name


@pytest.mark.parametrize(
    "tag, expected_tag",
    [
        ("....asdf.", "asdf."),
        ("A" * 128, "A" * 128),
        ("A" * 129, "A" * 128),
        ("-asdf-", "asdf-"),
        (".-asdf", "asdf"),
    ],
)
def test_to_valid_docker_image_version(tag, expected_tag):
    assert to_valid_docker_image_version(tag) == expected_tag


@pytest.mark.parametrize(
    "tag", ["randomtag", "name:version", "asdf123:" + "A" * 128, "a-a.a__a"]
)
def test_validate_tag(tag):
    # check to make sure tag is returned and nothing is raised
    assert validate_tag(None, None, tag) == tag


@pytest.mark.parametrize(
    "tag", ["AAA--", ".asdf", "asdf:...", "asdf:" + "A" * 129, "asdf:å"]
)
def test_validate_tag_raises(tag):
    with pytest.raises(click.BadParameter):
        validate_tag(None, None, tag)


def test_run_command_with_input_file(bento_bundle_path):
    input_path = generate_test_input_file()
    runner = CliRunner()

    cli = create_bento_service_cli()
    run_cmd = cli.commands["run"]
    result = runner.invoke(
        run_cmd,
        [
            bento_bundle_path,
            "predict_dataframe",
            "--input",
            input_path,
            "-o",
            "json",
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == '3'

    result = runner.invoke(
        run_cmd,
        [
            bento_bundle_path,
            "predict_dataframe_v1",
            "--input",
            input_path,
            "-o",
            "json",
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == '3'


def test_echo_docker_api_result_build():
    build_stream = [
        {'stream': 'Step 1/2 : FROM bentoml/model-server:0.8.1'},
        {'stream': '\n'},
        {'stream': ' ---> f034fa23264c\n'},
        {'stream': 'Step 2/2 : COPY . /bento'},
        {'stream': '\n'},
        {'stream': ' ---> Using cache\n'},
        {'aux': {'ID': 'sha256:garbagehash'}},
        {'stream': 'Successfully built 9d9918e008dd\n'},
        {'stream': 'Successfully tagged personal/test:latest\n'},
    ]

    expected = [
        "Step 1/2 : FROM bentoml/model-server:0.8.1",
        " ---> f034fa23264c",
        "Step 2/2 : COPY . /bento",
        " ---> Using cache",
        "Successfully built 9d9918e008dd",
        "Successfully tagged personal/test:latest",
    ]

    res = [line for line in echo_docker_api_result(build_stream)]
    assert_equal_lists(res, expected)


def test_echo_docker_api_result_push_no_access():
    push_stream = [
        {'status': 'The push refers to repository [docker.io/library/asdf]'},
        {'status': 'Preparing', 'progressDetail': {}, 'id': '2e280b8a5f3e'},
        {'status': 'Preparing', 'progressDetail': {}, 'id': 'd0b7e1b96cc1'},
        {'status': 'Preparing', 'progressDetail': {}, 'id': 'fcd8d39597dd'},
        {'status': 'Waiting', 'progressDetail': {}, 'id': '2e280b8a5f3e'},
        {'status': 'Waiting', 'progressDetail': {}, 'id': 'd0b7e1b96cc1'},
        {'status': 'Waiting', 'progressDetail': {}, 'id': 'fcd8d39597dd'},
        {
            'errorDetail': {
                'message': 'denied: requested access to the resource is denied'
            },
            'error': 'denied: requested access to the resource is denied',
        },
    ]

    with pytest.raises(BentoMLException) as e:
        _ = [line for line in echo_docker_api_result(push_stream)]
    assert str(e.value) == 'denied: requested access to the resource is denied'


def test_echo_docker_api_result_push():
    push_stream = [
        {'status': 'The push refers to repository [docker.io/personal/test]'},
        {'status': 'Preparing', 'progressDetail': {}, 'id': '2e280b8a5f3e'},
        {'status': 'Preparing', 'progressDetail': {}, 'id': '03613b6b1004'},
        {'status': 'Waiting', 'progressDetail': {}, 'id': 'cb2960c4c4d1'},
        {
            'status': 'Pushing',
            'progressDetail': {'current': 5632, 'total': 532223},
            'progress': '[=>    ]',
            'id': '03613b6b1004',
        },
        {
            'status': 'Pushing',
            'progressDetail': {'current': 512, 'total': 699},
            'progress': '[=====> ]',
            'id': '2e280b8a5f3e',
        },
        {
            'status': 'Pushing',
            'progressDetail': {'current': 128512, 'total': 532223},
            'progress': '[==>   ]',
            'id': '03613b6b1004',
        },
        {'status': 'Pushed', 'progressDetail': {}, 'id': '2e280b8a5f3e'},
        {'status': 'latest: digest: sha256:garbagehash size: 2214'},
        {
            'progressDetail': {},
            'aux': {'Tag': 'latest', 'Digest': 'sha256:garbagehash', 'Size': 2214},
        },
    ]

    expected = [
        f"Pushed {format_size(5632)} / {format_size(532223)}",
        f"Pushed {format_size(5632 + 512)} / {format_size(532223 + 699)}",
        f"Pushed {format_size(128512 + 512)} / {format_size(532223 + 699)}",
    ]

    res = [line for line in echo_docker_api_result(push_stream)]
    assert_equal_lists(res, expected)


@pytest.mark.skipif('not psutil.POSIX')
def test_gunicorn_serve_command():
    runner = CliRunner()

    cli = create_bento_service_cli()
    gunicorn_cmd = cli.commands["serve-gunicorn"]

    with mock.patch(
        'bentoml.server.gunicorn_server.GunicornBentoServer',
    ) as mocked_class:
        runner.invoke(
            gunicorn_cmd, ["/"],
        )
        instance = mocked_class.return_value
        assert instance.run.called
