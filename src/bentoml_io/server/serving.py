import os
import typing as t

import psutil
from simple_di import Provide
from simple_di import inject

from bentoml._internal.container import BentoMLContainer

from .app import ServiceAppFactory
from .service import Service


@inject
def serve_http(
    service: Service,
    *,
    host: str = Provide[BentoMLContainer.http.host],
    port: int = Provide[BentoMLContainer.http.port],
    fd: int | None = None,
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    timeout: int | None = None,
    ssl_certfile: str | None = None,
    ssl_keyfile: str | None = None,
    ssl_keyfile_password: str | None = None,
    ssl_version: int | None = None,
    ssl_cert_reqs: int | None = None,
    ssl_ca_certs: str | None = None,
    ssl_ciphers: str | None = None,
    prometheus_dir: str | None = None,
    reload: bool = False,
    is_main: bool = False,
    worker_id: int | None = None,
    development_mode: bool = False,
) -> None:
    import uvicorn

    from bentoml._internal.configuration.containers import BentoMLContainer
    from bentoml._internal.context import component_context
    from bentoml._internal.log import configure_server_logging

    component_context.component_type = "api_server"
    if worker_id is not None:
        component_context.component_index = worker_id

    configure_server_logging()
    BentoMLContainer.development_mode.set(development_mode)

    if prometheus_dir is not None:
        BentoMLContainer.prometheus_multiproc_dir.set(prometheus_dir)
    component_context.component_name = service.name

    app_factory = ServiceAppFactory(
        service,
        timeout=timeout,
        max_concurrency=BentoMLContainer.api_server_config.traffic.max_concurrency.get(),
        enable_metrics=BentoMLContainer.http.cors.enabled.get(),
    )
    asgi_app = app_factory(is_main=is_main)
    uvicorn_extra_options: dict[str, t.Any] = {}
    if fd is not None:
        uvicorn_extra_options["fd"] = fd
    else:
        uvicorn_extra_options.update(host=host, port=port)
    if ssl_version is not None:
        uvicorn_extra_options["ssl_version"] = ssl_version
    if ssl_cert_reqs is not None:
        uvicorn_extra_options["ssl_cert_reqs"] = ssl_cert_reqs
    if ssl_ciphers is not None:
        uvicorn_extra_options["ssl_ciphers"] = ssl_ciphers

    if psutil.WINDOWS:
        # 1. uvloop is not supported on Windows
        # 2. the default policy for Python > 3.8 on Windows is ProactorEventLoop, which doesn't
        #    support listen on a existing socket file descriptors
        # See https://docs.python.org/3.8/library/asyncio-platforms.html#windows
        uvicorn_extra_options["loop"] = "asyncio"
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
    uvicorn.run(
        app=asgi_app,
        backlog=backlog,
        log_config=None,
        workers=1,
        reload=reload,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ssl_ca_certs=ssl_ca_certs,
        **uvicorn_extra_options,
    )


@inject
def serve_http_production(
    bento_identifier: str | Service,
    working_dir: str,
    port: int = Provide[BentoMLContainer.http.port],
    host: str = Provide[BentoMLContainer.http.host],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    api_workers: int = Provide[BentoMLContainer.api_server_workers],
    timeout: int | None = None,
    ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
    ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
    ssl_keyfile_password: str | None = Provide[BentoMLContainer.ssl.keyfile_password],
    ssl_version: int | None = Provide[BentoMLContainer.ssl.version],
    ssl_cert_reqs: int | None = Provide[BentoMLContainer.ssl.cert_reqs],
    ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
    ssl_ciphers: str | None = Provide[BentoMLContainer.ssl.ciphers],
    bentoml_home: str = Provide[BentoMLContainer.bentoml_home],
    development_mode: bool = False,
    reload: bool = False,
) -> None:
    from bentoml._internal.service.loader import load
    from bentoml.serve import ensure_prometheus_dir

    prometheus_dir = ensure_prometheus_dir()
    working_dir = os.path.realpath(os.path.expanduser(working_dir))
    if isinstance(bento_identifier, Service):
        svc = bento_identifier
    else:
        svc = t.cast("Service", load(bento_identifier, working_dir))
    if development_mode:
        serve_http(
            svc,
            host=host,
            port=port,
            reload=reload,
            development_mode=True,
            timeout=timeout,
            prometheus_dir=prometheus_dir,
            is_main=True,
            backlog=backlog,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            ssl_keyfile_password=ssl_keyfile_password,
            ssl_version=ssl_version,
            ssl_cert_reqs=ssl_cert_reqs,
            ssl_ca_certs=ssl_ca_certs,
            ssl_ciphers=ssl_ciphers,
        )
        return
