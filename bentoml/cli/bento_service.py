import click
import os
import json
import re
import psutil

from bentoml import __version__
from bentoml.utils.lazy_loader import LazyLoader
from bentoml.utils.s3 import is_s3_url
from bentoml.server.api_server import BentoAPIServer
from bentoml.exceptions import BentoMLException, CLIException
from bentoml.server import start_dev_server, start_prod_server
from bentoml.server.open_api import get_open_api_spec_json
from bentoml.utils import (
    ProtoMessageToDict,
    status_pb_to_error_code_and_message,
)
from bentoml.cli.click_utils import (
    CLI_COLOR_WARNING,
    CLI_COLOR_SUCCESS,
    _echo,
    BentoMLCommandGroup,
    conditional_argument,
)
from bentoml.cli.utils import (
    echo_docker_api_result,
    Spinner,
    get_default_yatai_client,
)
from bentoml.saved_bundle import (
    load,
    load_bento_service_api,
    load_bento_service_metadata,
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


def escape_shell_params(param):
    k, v = param.split('=')
    v = re.sub(r'([^a-zA-Z0-9])', r'\\\1', v)
    return '{}={}'.format(k, v)


def to_valid_docker_image_name(name):
    # https://docs.docker.com/engine/reference/commandline/tag/#extended-description
    return name.lower().strip("._-")


def to_valid_docker_image_version(version):
    # https://docs.docker.com/engine/reference/commandline/tag/#extended-description
    return version.encode("ascii", errors="ignore").decode().lstrip(".-")[:128]


def validate_tag(ctx, param, tag):  # pylint: disable=unused-argument
    if tag is None:
        return tag

    if ":" in tag:
        name, version = tag.split(":")[:2]
    else:
        name, version = tag, None

    valid_name_pattern = re.compile(
        r"""
        ^(
        [a-z0-9]+      # alphanumeric
        (.|_{1,2}|-+)? # seperators
        )*$
        """,
        re.VERBOSE,
    )
    valid_version_pattern = re.compile(
        r"""
        ^
        [a-zA-Z0-9] # cant start with .-
        [ -~]{,127} # ascii match rest, cap at 128
        $
        """,
        re.VERBOSE,
    )

    if not valid_name_pattern.match(name):
        raise click.BadParameter(
            f"Provided Docker Image tag {tag} is invalid. "
            "Name components may contain lowercase letters, digits "
            "and separators. A separator is defined as a period, "
            "one or two underscores, or one or more dashes.",
            ctx=ctx,
            param=param,
        )
    if version and not valid_version_pattern.match(version):
        raise click.BadParameter(
            f"Provided Docker Image tag {tag} is invalid. "
            "A tag name must be valid ASCII and may contain "
            "lowercase and uppercase letters, digits, underscores, "
            "periods and dashes. A tag name may not start with a period "
            "or a dash and may contain a maximum of 128 characters.",
            ctx=ctx,
            param=param,
        )
    return tag


def resolve_bundle_path(bento, pip_installed_bundle_path):
    if pip_installed_bundle_path:
        assert (
            bento is None
        ), "pip installed BentoService commands should not have Bento argument"
        return pip_installed_bundle_path

    if os.path.isdir(bento) or is_s3_url(bento):
        # saved_bundle already support loading local and s3 path
        return bento

    elif ":" in bento:
        # assuming passing in BentoService in the form of Name:Version tag
        yatai_client = get_default_yatai_client()
        name, version = bento.split(':')
        get_bento_result = yatai_client.repository.get(name, version)
        if get_bento_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_bento_result.status
            )
            raise BentoMLException(
                f'BentoService {name}:{version} not found - '
                f'{error_code}:{error_message}'
            )
        if get_bento_result.bento.uri.s3_presigned_url:
            # Use s3 presigned URL for downloading the repository if it is presented
            return get_bento_result.bento.uri.s3_presigned_url
        else:
            return get_bento_result.bento.uri.uri
    else:
        raise BentoMLException(
            f'BentoService "{bento}" not found - either specify the file path of '
            f'the BentoService saved bundle, or the BentoService id in the form of '
            f'"name:version"'
        )


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
        saved_bundle_path = resolve_bundle_path(bento, pip_installed_bundle_path)

        api = load_bento_service_api(saved_bundle_path, api_name)
        api.handle_cli(run_args)

    # Example Usage: bentoml info {BUNDLE_PATH}
    @bentoml_cli.command(
        help="List all APIs defined in the BentoService loaded from saved bundle",
        short_help="List APIs",
    )
    @conditional_argument(pip_installed_bundle_path is None, "bento", type=click.STRING)
    def info(bento=None):
        """
        List all APIs defined in the BentoService loaded from saved bundle
        """
        saved_bundle_path = resolve_bundle_path(bento, pip_installed_bundle_path)

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
    def open_api_spec(bento=None):
        saved_bundle_path = resolve_bundle_path(bento, pip_installed_bundle_path)

        bento_service = load(saved_bundle_path)

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
        default=BentoAPIServer._DEFAULT_PORT,
        help=f"The port to listen on for the REST api server, "
        f"default is {BentoAPIServer._DEFAULT_PORT}",
        envvar='BENTOML_PORT',
    )
    @click.option(
        '--enable-microbatch/--disable-microbatch',
        default=False,
        help="Run API server with micro-batch enabled",
        envvar='BENTOML_ENABLE_MICROBATCH',
    )
    @click.option(
        '--run-with-ngrok',
        is_flag=True,
        default=False,
        help="Use ngrok to relay traffic on a public endpoint to this"
        "API server on localhost",
        envvar='BENTOML_ENABLE_NGROK',
    )
    def serve(port, bento=None, enable_microbatch=False, run_with_ngrok=False):
        saved_bundle_path = resolve_bundle_path(bento, pip_installed_bundle_path)
        start_dev_server(saved_bundle_path, port, enable_microbatch, run_with_ngrok)

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
        default=BentoAPIServer._DEFAULT_PORT,
        help=f"The port to listen on for the REST api server, "
        f"default is {BentoAPIServer._DEFAULT_PORT}",
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
    @click.option(
        '--enable-microbatch/--disable-microbatch',
        default=False,
        help="Run API server with micro batch enabled",
        envvar='BENTOML_ENABLE_MICROBATCH',
    )
    @click.option(
        '--microbatch-workers',
        type=click.INT,
        default=1,
        help="Number of micro-batch request dispatcher workers",
        envvar='BENTOML_MICROBATCH_WORKERS',
    )
    def serve_gunicorn(
        port,
        workers,
        timeout,
        bento=None,
        enable_microbatch=False,
        microbatch_workers=1,
    ):
        if not psutil.POSIX:
            _echo(
                "The `bentoml server-gunicon` command is only supported on POSIX. "
                "On windows platform, use `bentoml serve` for local API testing and "
                "docker for running production API endpoint: "
                "https://docs.docker.com/docker-for-windows/ "
            )
            return
        saved_bundle_path = resolve_bundle_path(bento, pip_installed_bundle_path)
        start_prod_server(
            saved_bundle_path,
            port,
            timeout,
            workers,
            enable_microbatch,
            microbatch_workers,
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
    @click.option('-p', '--push', is_flag=True)
    @click.option(
        '-t',
        '--tag',
        help="Optional image tag. If not specified, Bento will generate one from "
        "the name of the Bento.",
        required=False,
        callback=validate_tag,
    )
    @click.option(
        '-u', '--username', type=click.STRING, required=False,
    )
    @click.option(
        '-p', '--password', type=click.STRING, required=False,
    )
    def containerize(bento, push, tag, username, password):
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
        saved_bundle_path = resolve_bundle_path(bento, pip_installed_bundle_path)

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

        import docker

        docker_api = docker.APIClient()
        try:
            with Spinner(f"Building Docker image {tag} from {bento} \n"):
                for line in echo_docker_api_result(
                    docker_api.build(path=saved_bundle_path, tag=tag, decode=True,)
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
