import argparse
import json
import logging
import re
import sys
from typing import TYPE_CHECKING

import click
import psutil
from simple_di import Provide, inject

from bentoml import __version__
from bentoml.cli.click_utils import (
    BentoMLCommandGroup,
    CLI_COLOR_SUCCESS,
    _echo,
    conditional_argument,
)
from bentoml.cli.utils import Spinner
from bentoml.configuration.containers import BentoMLContainer
from bentoml.saved_bundle import (
    load_bento_service_api,
    load_bento_service_metadata,
    load_from_dir,
)
from bentoml.server import start_dev_server, start_prod_server
from bentoml.utils import ProtoMessageToDict, resolve_bundle_path
from bentoml.utils.docker_utils import validate_tag
from bentoml.utils.lazy_loader import LazyLoader
from bentoml.utils.open_api import get_open_api_spec_json
from bentoml.yatai.client import get_yatai_client

if TYPE_CHECKING:
    from bentoml.yatai.proto.repository_pb2 import BentoServiceMetadata
    from bentoml.yatai.client import YataiClient

try:
    import click_completion

    click_completion.init()
    shell_types = click_completion.DocumentedChoice(click_completion.core.shells)
except ImportError:
    # click_completion package is optional to use BentoML cli,
    click_completion = None
    shell_types = click.Choice(['bash', 'zsh', 'fish', 'powershell'])

logger = logging.getLogger(__name__)

yatai_proto = LazyLoader('yatai_proto', globals(), 'bentoml.yatai.proto')


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


def escape_shell_params(param):
    k, v = param.split("=")
    v = re.sub(r"([^a-zA-Z0-9])", r"\\\1", v)
    return "{}={}".format(k, v)


@inject
def create_bento_service_cli(
    pip_installed_bundle_path: str = None,
    default_yatai_url: str = None,
    default_port: int = Provide[BentoMLContainer.config.bento_server.port],
    default_mb_max_batch_size: int = Provide[
        BentoMLContainer.config.bento_server.microbatch.max_batch_size
    ],
    default_mb_max_latency: int = Provide[
        BentoMLContainer.config.bento_server.microbatch.max_latency
    ],
    default_run_with_ngrok: bool = Provide[
        BentoMLContainer.config.bento_server.ngrok.enabled
    ],
    default_enable_swagger: bool = Provide[
        BentoMLContainer.config.bento_server.swagger.enabled
    ],
    default_workers: int = Provide[BentoMLContainer.config.bento_server.workers],
    default_microbatch_workers: int = Provide[
        BentoMLContainer.config.bento_server.microbatch.workers
    ],
    default_timeout: int = Provide[BentoMLContainer.config.bento_server.timeout],
):
    # pylint: disable=unused-variable

    batch_options = [
        click.option(
            '--enable-microbatch/--disable-microbatch',
            default=None,
            help="Deprecated option for running API server with micro-batch enabled.",
            envvar='BENTOML_ENABLE_MICROBATCH',
        ),
        click.option(
            '--mb-max-batch-size',
            type=click.INT,
            help="Specify micro batching maximal batch size.",
            envvar='BENTOML_MB_MAX_BATCH_SIZE',
            default=default_mb_max_batch_size,
        ),
        click.option(
            '--mb-max-latency',
            type=click.INT,
            help="Specify micro batching maximal latency in milliseconds.",
            envvar='BENTOML_MB_MAX_LATENCY',
            default=default_mb_max_latency,
        ),
    ]

    @click.group(cls=BentoMLCommandGroup)
    @click.version_option(version=__version__)
    def bentoml_cli():
        """
        BentoML CLI tool
        """

    # Example Usage: bentoml run {API_NAME} {BUNDLE_PATH} --input ...
    @bentoml_cli.command(
        help="Run a API defined in saved BentoService bundle from command line",
        short_help="Run API function",
        context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
    )
    @conditional_argument(pip_installed_bundle_path is None, "bento", type=click.STRING)
    @click.argument("api_name", type=click.STRING)
    @click.argument('run_args', nargs=-1, type=click.UNPROCESSED)
    def run(api_name, run_args, bento=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--yatai-url', type=str, default=default_yatai_url)
        parsed_args, _ = parser.parse_known_args(run_args)
        yatai_url = parsed_args.yatai_url
        saved_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path, yatai_url
        )

        api = load_bento_service_api(saved_bundle_path, api_name)
        exit_code = api.handle_cli(run_args)
        sys.exit(exit_code)

    # Example Usage: bentoml info {BUNDLE_PATH}
    @bentoml_cli.command(
        help="List all APIs defined in the BentoService loaded from saved bundle",
        short_help="List APIs",
    )
    @conditional_argument(pip_installed_bundle_path is None, "bento", type=click.STRING)
    @click.option(
        '--yatai-url',
        type=click.STRING,
        default=default_yatai_url,
        help='Remote YataiService URL. Optional. '
        'Example: "--yatai-url http://localhost:50050"',
    )
    def info(bento=None, yatai_url=None):
        """
        List all APIs defined in the BentoService loaded from saved bundle
        """
        saved_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path, yatai_url
        )

        bento_service_metadata_pb = load_bento_service_metadata(saved_bundle_path)
        output = json.dumps(ProtoMessageToDict(bento_service_metadata_pb), indent=2)
        _echo(output)

    # Example usage: bentoml open-api-spec {BUNDLE_PATH}
    @bentoml_cli.command(
        name="open-api-spec",
        help="Display API specification JSON in Open-API format",
        short_help="Display OpenAPI/Swagger JSON specs",
    )
    @conditional_argument(pip_installed_bundle_path is None, "bento", type=click.STRING)
    @click.option(
        '--yatai-url',
        type=click.STRING,
        default=default_yatai_url,
        help='Remote YataiService URL. Optional. '
        'Example: "--yatai-url http://localhost:50050"',
    )
    def open_api_spec(bento=None, yatai_url=None):
        saved_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path, yatai_url
        )

        bento_service = load_from_dir(saved_bundle_path)

        _echo(json.dumps(get_open_api_spec_json(bento_service), indent=2))

    # Example Usage: bentoml serve {BUNDLE_PATH} --port={PORT}
    @bentoml_cli.command(
        help="Start a dev API server serving specified BentoService",
        short_help="Start local dev API server",
    )
    @conditional_argument(pip_installed_bundle_path is None, "bento", type=click.STRING)
    @click.option(
        "--port",
        type=click.INT,
        default=default_port,
        help="The port to listen on for the REST api server, default is 5000",
        envvar='BENTOML_PORT',
    )
    @add_options(batch_options)
    @click.option(
        '--run-with-ngrok',
        is_flag=True,
        default=default_run_with_ngrok,
        help="Use ngrok to relay traffic on a public endpoint to this "
        "API server on localhost",
        envvar='BENTOML_ENABLE_NGROK',
    )
    @click.option(
        '--yatai-url',
        type=click.STRING,
        default=default_yatai_url,
        help='Remote YataiService URL. Optional. '
        'Example: "--yatai-url http://localhost:50050"',
    )
    @click.option(
        '--enable-swagger/--disable-swagger',
        is_flag=True,
        default=default_enable_swagger,
        help="Run API server with Swagger UI enabled",
        envvar='BENTOML_ENABLE_SWAGGER',
    )
    def serve(
        port,
        bento,
        enable_microbatch,
        mb_max_batch_size,
        mb_max_latency,
        run_with_ngrok,
        yatai_url,
        enable_swagger,
    ):

        if enable_microbatch is not None:
            logger.warning(
                "Option --enable-microbatch/--disable-microbatch has been "
                "deprecated in the current release. The micro-batching option "
                "has become the default. Consider using --mb-max-batching=1 "
                "to simulate the effect of --disable-microbatch"
            )

        saved_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path, yatai_url
        )

        start_dev_server(
            saved_bundle_path,
            port=port,
            mb_max_batch_size=mb_max_batch_size,
            mb_max_latency=mb_max_latency,
            run_with_ngrok=run_with_ngrok,
            enable_swagger=enable_swagger,
        )

    # Example Usage:
    # bentoml serve-gunicorn {BUNDLE_PATH} --port={PORT} --workers={WORKERS}
    @bentoml_cli.command(
        help="Start a production API server serving specified BentoService",
        short_help="Start production API server",
    )
    @conditional_argument(pip_installed_bundle_path is None, "bento", type=click.STRING)
    @click.option(
        "-p",
        "--port",
        type=click.INT,
        default=default_port,
        help="The port to listen on for the REST api server, default is 5000",
        envvar='BENTOML_PORT',
    )
    @click.option(
        "-w",
        "--workers",
        type=click.INT,
        default=default_workers,
        help="Number of workers will start for the gunicorn server",
        envvar='BENTOML_GUNICORN_WORKERS',
    )
    @click.option(
        "--timeout",
        type=click.INT,
        default=default_timeout,
        help="Number of seconds for an API server request to timeout",
    )
    @add_options(batch_options)
    @click.option(
        '--microbatch-workers',
        type=click.INT,
        default=default_microbatch_workers,
        help="Number of micro-batch request dispatcher workers",
        envvar='BENTOML_MICROBATCH_WORKERS',
    )
    @click.option(
        '--yatai-url',
        type=click.STRING,
        default=default_yatai_url,
        help='Remote YataiService URL. Optional. '
        'Example: "--yatai-url http://localhost:50050"',
    )
    @click.option(
        '--enable-swagger/--disable-swagger',
        is_flag=True,
        default=default_enable_swagger,
        help="Run API server with Swagger UI enabled",
        envvar='BENTOML_ENABLE_SWAGGER',
    )
    def serve_gunicorn(
        port,
        workers,
        timeout,
        bento,
        enable_microbatch,
        mb_max_batch_size,
        mb_max_latency,
        microbatch_workers,
        yatai_url,
        enable_swagger,
    ):
        if not psutil.POSIX:
            _echo(
                "The `bentoml serve-gunicorn` command is only supported on POSIX. "
                "On windows platform, use `bentoml serve` for local API testing and "
                "docker for running production API endpoint: "
                "https://docs.docker.com/docker-for-windows/ "
            )
            return

        if enable_microbatch is not None:
            logger.warning(
                "Option --enable-microbatch/--disable-microbatch has been "
                "deprecated in the current release. The micro-batching option "
                "has become the default. Consider using --mb-max-batching=1 "
                "to simulate the effect of --disable-microbatch"
            )
        saved_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path, yatai_url
        )

        start_prod_server(
            saved_bundle_path,
            port=port,
            workers=workers,
            timeout=timeout,
            enable_swagger=enable_swagger,
            mb_max_batch_size=mb_max_batch_size,
            mb_max_latency=mb_max_latency,
            microbatch_workers=microbatch_workers,
        )

    @bentoml_cli.command(
        help="Install shell command completion",
        short_help="Install shell command completion",
    )
    @click.option(
        '--append/--overwrite',
        help="Append the completion code to the file",
        default=None,
    )
    @click.argument('shell', required=False, type=shell_types)
    @click.argument('path', required=False)
    def install_completion(append, shell, path):
        if click_completion:
            # click_completion package is imported
            shell, path = click_completion.core.install(
                shell=shell, path=path, append=append
            )
            click.echo('%s completion installed in %s' % (shell, path))
        else:
            click.echo(
                "'click_completion' is required for BentoML auto-completion, "
                "install it with `pip install click_completion`"
            )

    # bentoml containerize {service_name}:{service_tag} -t {docker_tags}
    @bentoml_cli.command(
        help='Containerizes given Bento into a ready-to-use Docker image.',
        short_help="Containerizes given Bento into a ready-to-use Docker image",
    )
    @click.argument("bento", type=click.STRING)
    @click.option('--push', is_flag=True)
    @click.option(
        '-t',
        '--tag',
        help="Optional image tag. If not specified, Bento will generate one from "
        "the name of the Bento.",
        required=False,
        callback=validate_tag,
    )
    @click.option(
        '--build-arg', multiple=True, help="pass through docker image build arguments"
    )
    @click.option(
        '--yatai-url',
        type=click.STRING,
        default=default_yatai_url,
        help='Specify the YataiService for running the containerization, default to '
        'the Local YataiService with local docker daemon. Example: '
        '"--yatai-url http://localhost:50050"',
    )
    def containerize(bento, push, tag, build_arg, yatai_url):
        """Containerize specified BentoService.

        BENTO is the target BentoService to be containerized, referenced by its name
        and version in format of name:version. For example: "iris_classifier:v1.2.0"

        `bentoml containerize` command also supports the use of the `latest` tag
        which will automatically use the last built version of your Bento.

        You can provide a tag for the image built by Bento using the
        `--tag` flag. Additionally, you can provide a `--push` flag,
        which will push the built image to the Docker repository specified by the
        image tag.

        You can also prefixing the tag with a hostname for the repository you wish
        to push to.
        e.g. `bentoml containerize IrisClassifier:latest --push --tag
        repo-address.com:username/iris` would build a Docker image called
        `username/iris:latest` and push that to docker repository at repo-address.com.

        By default, the `containerize` command will use the current credentials
        provided by Docker daemon.
        """
        saved_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path, yatai_url
        )

        _echo(f"Found Bento: {saved_bundle_path}")

        # fmt: off
        bento_metadata: "BentoServiceMetadata" = load_bento_service_metadata(saved_bundle_path)  # noqa: E501
        # fmt: on

        bento_tag = f'{bento_metadata.name}:{bento_metadata.version}'
        yatai_client: "YataiClient" = get_yatai_client(yatai_url)
        docker_build_args = {}
        if build_arg:
            for arg in build_arg:
                key, value = arg.split("=", 1)
                docker_build_args[key] = value
        if yatai_url is not None:
            spinner_message = f'Sending containerize RPC to YataiService at {yatai_url}'
        else:
            spinner_message = (
                f'Containerizing {bento_tag} with local YataiService and docker '
                f'daemon from local environment'
            )
        with Spinner(spinner_message):
            tag: str = yatai_client.repository.containerize(
                bento=bento_tag, tag=tag, build_args=docker_build_args, push=push,
            )
            _echo(f'\nBuild container image: {tag}', CLI_COLOR_SUCCESS)

    # pylint: enable=unused-variable
    return bentoml_cli
