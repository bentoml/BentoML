import argparse
import click
import sys

import json
import re
import psutil

from bentoml import __version__
from bentoml.utils.lazy_loader import LazyLoader
from bentoml.server.api_server import BentoAPIServer
from bentoml.exceptions import BentoMLException, CLIException
from bentoml.server import start_dev_server, start_prod_server
from bentoml.server.open_api import get_open_api_spec_json
from bentoml.utils import (
    ProtoMessageToDict,
    resolve_bundle_path,
)
from bentoml.cli.click_utils import (
    CLI_COLOR_WARNING,
    CLI_COLOR_SUCCESS,
    _echo,
    BentoMLCommandGroup,
    conditional_argument,
)
from bentoml.cli.utils import echo_docker_api_result, Spinner
from bentoml.saved_bundle import (
    load_from_dir,
    load_bento_service_api,
    load_bento_service_metadata,
)
from bentoml.utils.docker_utils import (
    validate_tag,
    to_valid_docker_image_name,
    to_valid_docker_image_version,
)

try:
    import click_completion

    click_completion.init()
    shell_types = click_completion.DocumentedChoice(click_completion.core.shells)
except ImportError:
    # click_completion package is optional to use BentoML cli,
    click_completion = None
    shell_types = click.Choice(['bash', 'zsh', 'fish', 'powershell'])


yatai_proto = LazyLoader('yatai_proto', globals(), 'bentoml.yatai.proto')

batch_options = [
    click.option(
        '--enable-microbatch/--disable-microbatch',
        default=False,
        help="Run API server with micro-batch enabled",
        envvar='BENTOML_ENABLE_MICROBATCH',
    ),
    click.option(
        '--mb-max-batch-size',
        type=click.INT,
        help="Specify micro batching maximal batch size.",
        envvar='BENTOML_MB_MAX_BATCH_SIZE',
    ),
    click.option(
        '--mb-max-latency',
        type=click.INT,
        help="Specify micro batching maximal latency in milliseconds.",
        envvar='BENTOML_MB_MAX_LATENCY',
    ),
]


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


def create_bento_service_cli(pip_installed_bundle_path=None):
    # pylint: disable=unused-variable

    @click.group(cls=BentoMLCommandGroup)
    @click.version_option(version=__version__)
    def bentoml_cli():
        """
        BentoML CLI tool
        """

    # Example Usage: bentoml run {API_NAME} {BUNDLE_PATH} --input=...
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
        parser.add_argument('--yatai-url', type=str, default=None)
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
        default=BentoAPIServer.DEFAULT_PORT,
        help=f"The port to listen on for the REST api server, "
        f"default is {BentoAPIServer.DEFAULT_PORT}",
        envvar='BENTOML_PORT',
    )
    @add_options(batch_options)
    @click.option(
        '--run-with-ngrok',
        is_flag=True,
        default=False,
        help="Use ngrok to relay traffic on a public endpoint to this "
        "API server on localhost",
        envvar='BENTOML_ENABLE_NGROK',
    )
    @click.option(
        '--yatai-url',
        type=click.STRING,
        help='Remote YataiService URL. Optional. '
        'Example: "--yatai-url http://localhost:50050"',
    )
    @click.option(
        '--enable-swagger/--disable-swagger',
        is_flag=True,
        default=True,
        help="Run API server with Swagger UI enabled",
        envvar='BENTOML_ENABLE_SWAGGER',
    )
    def serve(
        port,
        bento=None,
        enable_microbatch=False,
        mb_max_batch_size=None,
        mb_max_latency=None,
        run_with_ngrok=False,
        yatai_url=None,
        enable_swagger=True,
    ):
        saved_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path, yatai_url
        )
        start_dev_server(
            saved_bundle_path,
            port,
            enable_microbatch,
            mb_max_batch_size,
            mb_max_latency,
            run_with_ngrok,
            enable_swagger,
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
        default=BentoAPIServer.DEFAULT_PORT,
        help=f"The port to listen on for the REST api server, "
        f"default is {BentoAPIServer.DEFAULT_PORT}",
        envvar='BENTOML_PORT',
    )
    @click.option(
        "-w",
        "--workers",
        type=click.INT,
        default=None,
        help="Number of workers will start for the gunicorn server",
        envvar='BENTOML_GUNICORN_WORKERS',
    )
    @click.option("--timeout", type=click.INT, default=None)
    @add_options(batch_options)
    @click.option(
        '--microbatch-workers',
        type=click.INT,
        default=1,
        help="Number of micro-batch request dispatcher workers",
        envvar='BENTOML_MICROBATCH_WORKERS',
    )
    @click.option(
        '--yatai-url',
        type=click.STRING,
        help='Remote YataiService URL. Optional. '
        'Example: "--yatai-url http://localhost:50050"',
    )
    @click.option(
        '--enable-swagger/--disable-swagger',
        is_flag=True,
        default=True,
        help="Run API server with Swagger UI enabled",
        envvar='BENTOML_ENABLE_SWAGGER',
    )
    def serve_gunicorn(
        port,
        workers,
        timeout,
        bento=None,
        enable_microbatch=False,
        mb_max_batch_size=None,
        mb_max_latency=None,
        microbatch_workers=1,
        yatai_url=None,
        enable_swagger=True,
    ):
        if not psutil.POSIX:
            _echo(
                "The `bentoml serve-gunicorn` command is only supported on POSIX. "
                "On windows platform, use `bentoml serve` for local API testing and "
                "docker for running production API endpoint: "
                "https://docs.docker.com/docker-for-windows/ "
            )
            return
        saved_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path, yatai_url
        )
        start_prod_server(
            saved_bundle_path,
            port,
            timeout,
            workers,
            enable_microbatch,
            mb_max_batch_size,
            mb_max_latency,
            microbatch_workers,
            enable_swagger,
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
        '-u', '--username', type=click.STRING, required=False,
    )
    @click.option(
        '-p', '--password', type=click.STRING, required=False,
    )
    @click.option(
        '--yatai-url',
        type=click.STRING,
        help='Remote YataiService URL. Optional. '
        'Example: "--yatai-url http://localhost:50050"',
    )
    def containerize(bento, push, tag, build_arg, username, password, yatai_url):
        """Containerize specified BentoService.

        BENTO is the target BentoService to be containerized, referenced by its name
        and version in format of name:version. For example: "iris_classifier:v1.2.0"

        `bentoml containerize` command also supports the use of the `latest` tag
        which will automatically use the last built version of your Bento.

        You can provide a tag for the image built by Bento using the
        `--docker-image-tag` flag. Additionally, you can provide a `--push` flag,
        which will push the built image to the Docker repository specified by the
        image tag.

        You can also prefixing the tag with a hostname for the repository you wish
        to push to.
        e.g. `bentoml containerize IrisClassifier:latest --push --tag username/iris`
        would build a Docker image called `username/iris:latest` and push that to
        Docker Hub.

        By default, the `containerize` command will use the credentials provided by
        Docker. You may provide your own through `--username` and `--password`.
        """
        saved_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path, yatai_url
        )

        _echo(f"Found Bento: {saved_bundle_path}")

        bento_metadata = load_bento_service_metadata(saved_bundle_path)
        name = to_valid_docker_image_name(bento_metadata.name)
        version = to_valid_docker_image_version(bento_metadata.version)

        if not tag:
            _echo(
                "Tag not specified, using tag parsed from "
                f"BentoService: '{name}:{version}'"
            )
            tag = f"{name}:{version}"
        if ":" not in tag:
            _echo(
                "Image version not specified, using version parsed "
                f"from BentoService: '{version}'",
                CLI_COLOR_WARNING,
            )
            tag = f"{tag}:{version}"

        docker_build_args = {}
        if build_arg:
            for arg in build_arg:
                key, value = arg.split("=")
                docker_build_args[key] = value

        import docker

        docker_api = docker.from_env().api
        try:
            with Spinner(f"Building Docker image {tag} from {bento} \n"):
                for line in echo_docker_api_result(
                    docker_api.build(
                        path=saved_bundle_path,
                        tag=tag,
                        decode=True,
                        buildargs=docker_build_args,
                    )
                ):
                    _echo(line)
        except docker.errors.APIError as error:
            raise CLIException(f'Could not build Docker image: {error}')

        _echo(
            f'Finished building {tag} from {bento}', CLI_COLOR_SUCCESS,
        )

        if push:
            auth_config_payload = (
                {"username": username, "password": password}
                if username or password
                else None
            )

            try:
                with Spinner(f"Pushing docker image to {tag}\n"):
                    for line in echo_docker_api_result(
                        docker_api.push(
                            repository=tag,
                            stream=True,
                            decode=True,
                            auth_config=auth_config_payload,
                        )
                    ):
                        _echo(line)
                _echo(
                    f'Pushed {tag} to {name}', CLI_COLOR_SUCCESS,
                )
            except (docker.errors.APIError, BentoMLException) as error:
                raise CLIException(f'Could not push Docker image: {error}')

    # pylint: enable=unused-variable
    return bentoml_cli
