def make_bento_name_docker_compatible(name, tag):
    """
    Name components may contain lowercase letters, digits and separators.
    A separator is defined as a period, one or two underscores, or one or more dashes.

    A tag name must be valid ASCII and may contain lowercase and uppercase letters,
    digits, underscores, periods and dashes. A tag name may not start with a period
    or a dash and may contain a maximum of 128 characters.

    https://docs.docker.com/engine/reference/commandline/tag/#extended-description
    """
    name = name.lower().strip("._-")
    tag = tag.lstrip(".-")[:128]
    return name, tag

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

    # pylint: enable=unused-variable
    return bentoml_cli
