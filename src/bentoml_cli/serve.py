from __future__ import annotations

import logging
import os
import sys
import typing as t

import click

if t.TYPE_CHECKING:
    P = t.ParamSpec("P")
    F = t.Callable[P, t.Any]


logger = logging.getLogger(__name__)

DEFAULT_DEV_SERVER_HOST = "127.0.0.1"


def deprecated_option(*param_decls: str, **attrs: t.Any):
    """Marks a given options as deprecated, and omit a warning when it's used"""
    deprecated = attrs.pop("deprecated", True)
    new_behaviour = attrs.pop("current_behaviour", None)
    assert new_behaviour is not None, "current_behaviour is required"

    def show_deprecated_callback(
        ctx: click.Context, param: click.Parameter, value: t.Any
    ):
        if value is not param.default and deprecated:
            name = "'--%(name)s'" if attrs.get("is_flag", False) else "'%(name)s'"
            DEPRECATION_WARNING = f"DeprecationWarning: The parameter {name} is deprecated and will be removed in the future. (Current behaviour: %(new_behaviour)s)"
            click.secho(
                DEPRECATION_WARNING
                % {"name": param.name, "new_behaviour": new_behaviour},
                fg="yellow",
                err=True,
            )

    def decorator(f: F[t.Any]) -> t.Callable[[F[t.Any]], click.Command]:
        msg = attrs.pop("help", "")
        msg += " (Deprecated)" if msg else "(Deprecated)"
        attrs.setdefault("help", msg)
        attrs.setdefault("callback", show_deprecated_callback)
        return click.option(*param_decls, **attrs)(f)

    return decorator


def add_serve_command(cli: click.Group) -> None:
    from bentoml._internal.configuration.containers import BentoMLContainer
    from bentoml._internal.log import configure_server_logging
    from bentoml.grpc.utils import LATEST_PROTOCOL_VERSION
    from bentoml_cli.env_manager import env_manager

    @cli.command(aliases=["serve-http"])
    @click.argument("bento", type=click.STRING, default=".")
    @click.option(
        "--development",
        type=click.BOOL,
        help="Run the BentoServer in development mode",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @deprecated_option(
        "--production",
        type=click.BOOL,
        help="Run BentoServer in production mode",
        current_behaviour="This is enabled by default. To run in development mode, use '--development'.",
        is_flag=True,
        default=True,
        show_default=False,
    )
    @click.option(
        "-p",
        "--port",
        type=click.INT,
        default=BentoMLContainer.http.port.get(),
        help="The port to listen on for the REST api server",
        envvar="BENTOML_PORT",
        show_default=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        default=BentoMLContainer.http.host.get(),
        help="The host to bind for the REST api server",
        envvar="BENTOML_HOST",
        show_default=True,
    )
    @click.option(
        "--api-workers",
        type=click.INT,
        default=BentoMLContainer.api_server_workers.get(),
        help="Specify the number of API server workers to start. Default to number of available CPU cores in production mode",
        envvar="BENTOML_API_WORKERS",
        show_default=True,
    )
    @click.option(
        "--timeout",
        type=click.INT,
        help="Specify the timeout (seconds) for API server and runners",
        envvar="BENTOML_TIMEOUT",
    )
    @click.option(
        "--backlog",
        type=click.INT,
        default=BentoMLContainer.api_server_config.backlog.get(),
        help="The maximum number of pending connections.",
        show_default=True,
    )
    @click.option(
        "--reload",
        type=click.BOOL,
        is_flag=True,
        help="Reload Service when code changes detected",
        default=False,
        show_default=True,
    )
    @click.option(
        "--working-dir",
        type=click.Path(),
        help="When loading from source code, specify the directory to find the Service instance",
        default=None,
        show_default=True,
    )
    @click.option(
        "--ssl-certfile",
        type=str,
        default=BentoMLContainer.ssl.certfile.get(),
        help="SSL certificate file",
        show_default=True,
    )
    @click.option(
        "--ssl-keyfile",
        type=str,
        default=BentoMLContainer.ssl.keyfile.get(),
        help="SSL key file",
        show_default=True,
    )
    @click.option(
        "--ssl-keyfile-password",
        type=str,
        default=BentoMLContainer.ssl.keyfile_password.get(),
        help="SSL keyfile password",
        show_default=True,
    )
    @click.option(
        "--ssl-version",
        type=int,
        default=BentoMLContainer.ssl.version.get(),
        help="SSL version to use (see stdlib 'ssl' module)",
        show_default=True,
    )
    @click.option(
        "--ssl-cert-reqs",
        type=int,
        default=BentoMLContainer.ssl.cert_reqs.get(),
        help="Whether client certificate is required (see stdlib 'ssl' module)",
        show_default=True,
    )
    @click.option(
        "--ssl-ca-certs",
        type=str,
        default=BentoMLContainer.ssl.ca_certs.get(),
        help="CA certificates file",
        show_default=True,
    )
    @click.option(
        "--ssl-ciphers",
        type=str,
        default=BentoMLContainer.ssl.ciphers.get(),
        help="Ciphers to use (see stdlib 'ssl' module)",
        show_default=True,
    )
    @env_manager
    def serve(  # type: ignore (unused warning)
        bento: str,
        development: bool,
        port: int,
        host: str,
        api_workers: int,
        timeout: int | None,
        backlog: int,
        reload: bool,
        working_dir: str | None,
        ssl_certfile: str | None,
        ssl_keyfile: str | None,
        ssl_keyfile_password: str | None,
        ssl_version: int | None,
        ssl_cert_reqs: int | None,
        ssl_ca_certs: str | None,
        ssl_ciphers: str | None,
        **attrs: t.Any,
    ) -> None:
        """Start a HTTP BentoServer from a given 🍱

        \b
        BENTO is the serving target, it can be the import as:
        - the import path of a 'bentoml.Service' instance
        - a tag to a Bento in local Bento store
        - a folder containing a valid 'bentofile.yaml' build file with a 'service' field, which provides the import path of a 'bentoml.Service' instance
        - a path to a built Bento (for internal & debug use only)

        e.g.:

        \b
        Serve from a bentoml.Service instance source code (for development use only):
            'bentoml serve fraud_detector.py:svc'

        \b
        Serve from a Bento built in local store:
            'bentoml serve fraud_detector:4tht2icroji6zput3suqi5nl2'
            'bentoml serve fraud_detector:latest'

        \b
        Serve from a Bento directory:
            'bentoml serve ./fraud_detector_bento'

        \b
        If '--reload' is provided, BentoML will detect code and model store changes during development, and restarts the service automatically.

        \b
        The '--reload' flag will:
        - be default, all file changes under '--working-dir' (default to current directory) will trigger a restart
        - when specified, respect 'include' and 'exclude' under 'bentofile.yaml' as well as the '.bentoignore' file in '--working-dir', for code and file changes
        - all model store changes will also trigger a restart (new model saved or existing model removed)
        """
        from bentoml import Service
        from bentoml._internal.service.loader import load

        configure_server_logging()
        if working_dir is None:
            if os.path.isdir(os.path.expanduser(bento)):
                working_dir = os.path.expanduser(bento)
            else:
                working_dir = "."
        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)
        svc = load(bento_identifier=bento, working_dir=working_dir)
        if isinstance(svc, Service):
            from bentoml.serve import serve_http_production

            if development:
                serve_http_production(
                    bento,
                    working_dir=working_dir,
                    port=port,
                    host=DEFAULT_DEV_SERVER_HOST if not host else host,
                    backlog=backlog,
                    api_workers=1,
                    timeout=timeout,
                    ssl_keyfile=ssl_keyfile,
                    ssl_certfile=ssl_certfile,
                    ssl_keyfile_password=ssl_keyfile_password,
                    ssl_version=ssl_version,
                    ssl_cert_reqs=ssl_cert_reqs,
                    ssl_ca_certs=ssl_ca_certs,
                    ssl_ciphers=ssl_ciphers,
                    reload=reload,
                    development_mode=True,
                )
            else:
                serve_http_production(
                    bento,
                    working_dir=working_dir,
                    port=port,
                    host=host,
                    api_workers=api_workers,
                    timeout=timeout,
                    ssl_keyfile=ssl_keyfile,
                    ssl_certfile=ssl_certfile,
                    ssl_keyfile_password=ssl_keyfile_password,
                    ssl_version=ssl_version,
                    ssl_cert_reqs=ssl_cert_reqs,
                    ssl_ca_certs=ssl_ca_certs,
                    ssl_ciphers=ssl_ciphers,
                    reload=reload,
                    development_mode=False,
                )
        else:
            from _bentoml_impl.server import serve_http

            svc.inject_config()
            serve_http(
                bento,
                working_dir=working_dir,
                host=host,
                port=port,
                backlog=backlog,
                timeout=timeout,
                ssl_certfile=ssl_certfile,
                ssl_keyfile=ssl_keyfile,
                ssl_keyfile_password=ssl_keyfile_password,
                ssl_version=ssl_version,
                ssl_cert_reqs=ssl_cert_reqs,
                ssl_ca_certs=ssl_ca_certs,
                ssl_ciphers=ssl_ciphers,
                development_mode=development,
                reload=reload,
            )

    @cli.command(name="serve-grpc")
    @click.argument("bento", type=click.STRING, default=".")
    @click.option(
        "--development",
        type=click.BOOL,
        help="Run the BentoServer in development mode",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @deprecated_option(
        "--production",
        type=click.BOOL,
        help="Run BentoServer in production mode",
        current_behaviour="This is enabled by default. To run in development mode, use '--development'.",
        is_flag=True,
        default=True,
        show_default=False,
    )
    @click.option(
        "-p",
        "--port",
        type=click.INT,
        default=BentoMLContainer.grpc.port.get(),
        help="The port to listen on for the REST api server",
        envvar="BENTOML_PORT",
        show_default=True,
    )
    @click.option(
        "--host",
        type=click.STRING,
        default=BentoMLContainer.grpc.host.get(),
        help="The host to bind for the gRPC server",
        envvar="BENTOML_HOST",
        show_default=True,
    )
    @click.option(
        "--api-workers",
        type=click.INT,
        default=BentoMLContainer.api_server_workers.get(),
        help="Specify the number of API server workers to start. Default to number of available CPU cores in production mode",
        envvar="BENTOML_API_WORKERS",
        show_default=True,
    )
    @click.option(
        "--reload",
        type=click.BOOL,
        is_flag=True,
        help="Reload Service when code changes detected",
        default=False,
        show_default=True,
    )
    @click.option(
        "--backlog",
        type=click.INT,
        default=BentoMLContainer.api_server_config.backlog.get(),
        help="The maximum number of pending connections.",
        show_default=True,
    )
    @click.option(
        "--working-dir",
        type=click.Path(),
        help="When loading from source code, specify the directory to find the Service instance",
        default=None,
        show_default=True,
    )
    @click.option(
        "--enable-reflection",
        is_flag=True,
        default=BentoMLContainer.grpc.reflection.enabled.get(),
        type=click.BOOL,
        help="Enable reflection.",
        show_default=True,
    )
    @click.option(
        "--enable-channelz",
        is_flag=True,
        default=BentoMLContainer.grpc.channelz.enabled.get(),
        type=click.BOOL,
        help="Enable Channelz. See https://github.com/grpc/proposal/blob/master/A14-channelz.md.",
    )
    @click.option(
        "--max-concurrent-streams",
        default=BentoMLContainer.grpc.max_concurrent_streams.get(),
        type=click.INT,
        help="Maximum number of concurrent incoming streams to allow on a http2 connection.",
        show_default=True,
    )
    @click.option(
        "--ssl-certfile",
        type=str,
        default=BentoMLContainer.ssl.certfile.get(),
        help="SSL certificate file",
        show_default=True,
    )
    @click.option(
        "--ssl-keyfile",
        type=str,
        default=BentoMLContainer.ssl.keyfile.get(),
        help="SSL key file",
        show_default=True,
    )
    @click.option(
        "--ssl-ca-certs",
        type=str,
        default=BentoMLContainer.ssl.ca_certs.get(),
        help="CA certificates file",
        show_default=True,
    )
    @click.option(
        "-pv",
        "--protocol-version",
        type=click.Choice(["v1", "v1alpha1"]),
        help="Determine the version of generated gRPC stubs to use.",
        default=LATEST_PROTOCOL_VERSION,
        show_default=True,
    )
    @env_manager
    def serve_grpc(  # type: ignore (unused warning)
        bento: str,
        development: bool,
        port: int,
        host: str,
        api_workers: int | None,
        backlog: int,
        reload: bool,
        working_dir: str | None,
        ssl_certfile: str | None,
        ssl_keyfile: str | None,
        ssl_ca_certs: str | None,
        enable_reflection: bool,
        enable_channelz: bool,
        max_concurrent_streams: int | None,
        protocol_version: str,
        **attrs: t.Any,
    ):
        """Start a gRPC BentoServer from a given 🍱

        \b
        BENTO is the serving target, it can be the import as:
        - the import path of a 'bentoml.Service' instance
        - a tag to a Bento in local Bento store
        - a folder containing a valid 'bentofile.yaml' build file with a 'service' field, which provides the import path of a 'bentoml.Service' instance
        - a path to a built Bento (for internal & debug use only)

        e.g.:

        \b
        Serve from a bentoml.Service instance source code (for development use only):
            'bentoml serve-grpc fraud_detector.py:svc'

        \b
        Serve from a Bento built in local store:
            'bentoml serve-grpc fraud_detector:4tht2icroji6zput3suqi5nl2'
            'bentoml serve-grpc fraud_detector:latest'

        \b
        Serve from a Bento directory:
            'bentoml serve-grpc ./fraud_detector_bento'

        If '--reload' is provided, BentoML will detect code and model store changes during development, and restarts the service automatically.

        \b
        The '--reload' flag will:
        - be default, all file changes under '--working-dir' (default to current directory) will trigger a restart
        - when specified, respect 'include' and 'exclude' under 'bentofile.yaml' as well as the '.bentoignore' file in '--working-dir', for code and file changes
        - all model store changes will also trigger a restart (new model saved or existing model removed)
        """
        configure_server_logging()
        if working_dir is None:
            if os.path.isdir(os.path.expanduser(bento)):
                working_dir = os.path.expanduser(bento)
            else:
                working_dir = "."

        from bentoml.serve import serve_grpc_production

        if development:
            serve_grpc_production(
                bento,
                working_dir=working_dir,
                port=port,
                host=DEFAULT_DEV_SERVER_HOST if not host else host,
                backlog=backlog,
                api_workers=1,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                ssl_ca_certs=ssl_ca_certs,
                max_concurrent_streams=max_concurrent_streams,
                reflection=enable_reflection,
                channelz=enable_channelz,
                protocol_version=protocol_version,
                reload=reload,
                development_mode=True,
            )
        else:
            serve_grpc_production(
                bento,
                working_dir=working_dir,
                port=port,
                backlog=backlog,
                host=host,
                api_workers=api_workers,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                ssl_ca_certs=ssl_ca_certs,
                max_concurrent_streams=max_concurrent_streams,
                reflection=enable_reflection,
                channelz=enable_channelz,
                protocol_version=protocol_version,
                reload=reload,
                development_mode=False,
            )
