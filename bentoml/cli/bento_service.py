import docker
import click
import os
import json
import re
import multiprocessing
import tempfile
import subprocess
import psutil

from pathlib import Path
from bentoml.utils.s3 import is_s3_url
from bentoml.server import BentoAPIServer
from bentoml.yatai.client import YataiClient
from bentoml.yatai.proto import status_pb2
from bentoml.exceptions import BentoMLException, CLIException
from ruamel.yaml import YAML
from bentoml.server.utils import get_gunicorn_num_of_workers
from bentoml.server.open_api import get_open_api_spec_json
from bentoml.marshal import MarshalService
from bentoml.utils import (
    ProtoMessageToDict,
    reserve_free_port,
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
)
from bentoml.saved_bundle import (
    load,
    load_bento_service_api,
    load_bento_service_metadata,
    load_saved_bundle_config,
)

try:
    import click_completion

    click_completion.init()
    shell_types = click_completion.DocumentedChoice(click_completion.core.shells)
except ImportError:
    # click_completion package is optional to use BentoML cli,
    click_completion = None
    shell_types = click.Choice(['bash', 'zsh', 'fish', 'powershell'])


def escape_shell_params(param):
    k, v = param.split('=')
    v = re.sub(r'([^a-zA-Z0-9])', r'\\\1', v)
    return '{}={}'.format(k, v)


def run_with_conda_env(bundle_path, command):
    config = load_saved_bundle_config(bundle_path)
    metadata = config['metadata']
    env_name = metadata['service_name'] + '_' + metadata['service_version']

    yaml = YAML()
    yaml.default_flow_style = False
    tmpf = tempfile.NamedTemporaryFile(delete=False)
    env_path = tmpf.name + '.yaml'
    yaml.dump(config['env']['conda_env'], Path(env_path))

    pip_req = os.path.join(bundle_path, 'requirements.txt')

    subprocess.call(
        'command -v conda >/dev/null 2>&1 || {{ echo >&2 "--with-conda '
        'parameter requires conda but it\'s not installed."; exit 1; }} && '
        'conda env update -n {env_name} -f {env_file} && '
        'conda init bash && '
        'eval "$(conda shell.bash hook)" && '
        'conda activate {env_name} && '
        '{{ [ -f {pip_req} ] && pip install -r {pip_req} || echo "no pip '
        'dependencies."; }} && {cmd}'.format(
            env_name=env_name, env_file=env_path, pip_req=pip_req, cmd=command,
        ),
        shell=True,
    )
    return


def make_bento_name_docker_compatible(name, version):
    """
    Name components may contain lowercase letters, digits and separators.
    A separator is defined as a period, one or two underscores, or one or more dashes.

    A tag name (version) must be valid ASCII and may contain lowercase and uppercase
    letters, digits, underscores, periods and dashes. A tag name may not start with
    a period or a dash and may contain a maximum of 128 characters.

    https://docs.docker.com/engine/reference/commandline/tag/#extended-description
    """
    name = name.lower().strip("._-")
    version = version.lstrip(".-")[:128]
    return name, version


def validate_tag(ctx, param, tag):  # pylint: disable=unused-argument
    if ":" in tag:
        tag = tag.split(":")[:2]
    else:
        _echo(
            f"Tag {tag} does not specify an image version, " f"using 'latest'",
            CLI_COLOR_WARNING,
        )
        tag = tag, "latest"
    name, version = make_bento_name_docker_compatible(*tag)
    return f"{name}:{version}"


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
        yatai_client = YataiClient()
        name, version = bento.split(':')
        get_bento_result = yatai_client.repository.get(name, version)
        if get_bento_result.status.status_code != status_pb2.Status.OK:
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
    @click.version_option()
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
    @click.option(
        '--with-conda',
        is_flag=True,
        default=False,
        help="Run API server in a BentoML managed Conda environment",
    )
    def run(api_name, run_args, bento=None, with_conda=False):
        bento_service_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path
        )

        if with_conda:
            return run_with_conda_env(
                bento_service_bundle_path,
                'bentoml run {api_name} {bento} {args}'.format(
                    bento=bento_service_bundle_path,
                    api_name=api_name,
                    args=' '.join(map(escape_shell_params, run_args)),
                ),
            )

        api = load_bento_service_api(bento_service_bundle_path, api_name)
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
        bento_service_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path
        )

        bento_service_metadata_pb = load_bento_service_metadata(
            bento_service_bundle_path
        )
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
        bento_service_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path
        )

        bento_service = load(bento_service_bundle_path)

        _echo(json.dumps(get_open_api_spec_json(bento_service), indent=2))

    # Example Usage: bentoml serve {BUNDLE_PATH} --port={PORT}
    @bentoml_cli.command(
        help="Start REST API server hosting BentoService loaded from saved bundle",
        short_help="Start local rest server",
    )
    @conditional_argument(pip_installed_bundle_path is None, "bento", type=click.STRING)
    @click.option(
        "--port",
        type=click.INT,
        default=BentoAPIServer._DEFAULT_PORT,
        help=f"The port to listen on for the REST api server, "
        f"default is ${BentoAPIServer._DEFAULT_PORT}",
        envvar='BENTOML_PORT',
    )
    @click.option(
        '--with-conda',
        is_flag=True,
        default=False,
        help="Run API server in a BentoML managed Conda environment",
    )
    @click.option(
        '--enable-microbatch',
        is_flag=True,
        default=False,
        help="(Beta) Run API server with micro-batch enabled",
        envvar='BENTOML_ENABLE_MICROBATCH',
    )
    def serve(port, bento=None, with_conda=False, enable_microbatch=False):
        bento_service_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path
        )
        bento_service = load(bento_service_bundle_path)

        if with_conda:
            return run_with_conda_env(
                bento_service_bundle_path,
                'bentoml serve {bento} --port {port} {flags}'.format(
                    bento=bento_service_bundle_path,
                    port=port,
                    flags="--enable-microbatch" if enable_microbatch else "",
                ),
            )

        if enable_microbatch:
            with reserve_free_port() as api_server_port:
                # start server right after port released
                #  to reduce potential race
                marshal_server = MarshalService(
                    bento_service_bundle_path,
                    outbound_host="localhost",
                    outbound_port=api_server_port,
                    outbound_workers=1,
                )
                api_server = BentoAPIServer(bento_service, port=api_server_port)
            marshal_server.async_start(port=port)
            api_server.start()
        else:
            api_server = BentoAPIServer(bento_service, port=port)
            api_server.start()

    # Example Usage:
    # bentoml serve-gunicorn {BUNDLE_PATH} --port={PORT} --workers={WORKERS}
    @bentoml_cli.command(
        help="Start REST API server from saved BentoService bundle with gunicorn",
        short_help="Start local gunicorn server",
    )
    @conditional_argument(pip_installed_bundle_path is None, "bento", type=click.STRING)
    @click.option(
        "-p",
        "--port",
        type=click.INT,
        default=BentoAPIServer._DEFAULT_PORT,
        help=f"The port to listen on for the REST api server, "
        f"default is ${BentoAPIServer._DEFAULT_PORT}",
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
        '--with-conda',
        is_flag=True,
        default=False,
        help="Run API server in a BentoML managed Conda environment",
    )
    @click.option(
        '--enable-microbatch',
        is_flag=True,
        default=False,
        help="(Beta) Run API server with micro batch enabled",
        envvar='BENTOML_ENABLE_MICROBATCH',
    )
    @click.option(
        '--microbatch-workers',
        type=click.INT,
        default=1,
        help="(Beta) Number of micro-batch request dispatcher workers",
        envvar='BENTOML_MICROBATCH_WORKERS',
    )
    def serve_gunicorn(
        port,
        workers,
        timeout,
        bento=None,
        with_conda=False,
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
        bento_service_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path
        )

        if with_conda:
            return run_with_conda_env(
                pip_installed_bundle_path,
                'bentoml serve_gunicorn {bento} -p {port} -w {workers} '
                '--timeout {timeout} {flags}'.format(
                    bento=bento_service_bundle_path,
                    port=port,
                    workers=workers,
                    timeout=timeout,
                    flags="--enable-microbatch" if enable_microbatch else "",
                ),
            )

        if workers is None:
            workers = get_gunicorn_num_of_workers()

        # Gunicorn only supports POSIX platforms
        from bentoml.server.gunicorn_server import GunicornBentoServer
        from bentoml.server.marshal_server import GunicornMarshalServer

        if enable_microbatch:
            prometheus_lock = multiprocessing.Lock()
            # avoid load model before gunicorn fork
            with reserve_free_port() as api_server_port:
                marshal_server = GunicornMarshalServer(
                    bundle_path=bento_service_bundle_path,
                    port=port,
                    workers=microbatch_workers,
                    prometheus_lock=prometheus_lock,
                    outbound_host="localhost",
                    outbound_port=api_server_port,
                    outbound_workers=workers,
                )

                gunicorn_app = GunicornBentoServer(
                    bento_service_bundle_path,
                    api_server_port,
                    workers,
                    timeout,
                    prometheus_lock,
                )
            marshal_server.async_run()
            gunicorn_app.run()
        else:
            gunicorn_app = GunicornBentoServer(
                bento_service_bundle_path, port, workers, timeout
            )
            gunicorn_app.run()

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
        bento_service_bundle_path = resolve_bundle_path(
            bento, pip_installed_bundle_path
        )
        _echo(f"Found Bento: {bento_service_bundle_path}")

        bento_service_metadata_pb = load_bento_service_metadata(
            bento_service_bundle_path
        )
        name, version = make_bento_name_docker_compatible(
            bento_service_metadata_pb.name, bento_service_metadata_pb.version,
        )

        # build docker compatible tag if one isnt provided
        if tag is None:
            tag = f"{name}:{version}"
        if tag != bento:
            _echo(
                f'Bento tag was changed to be Docker compatible. \n'
                f'"{bento}" -> "{tag}"',
                CLI_COLOR_WARNING,
            )

        docker_api = docker.APIClient()
        try:
            with Spinner(f"Building Docker image: {name}\n"):
                for line in echo_docker_api_result(
                    docker_api.build(
                        path=bento_service_bundle_path, tag=tag, decode=True,
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
                            repository=name,
                            tag=version,
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
