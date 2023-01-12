from __future__ import annotations

import os
import typing as t
import asyncio
import logging
import functools
import contextlib
import urllib.parse

import fs
import attr
import fs.errors
import fs.opener
import fs.opener.errors

from ..utils import LazyLoader
from ..runner.utils import Params
from ..runner.utils import Payload
from ..runner.runner import Runner
from ..runner.runner import RunnerMethod
from ..runner.runnable import RunnableMethodConfig
from ..runner.runner_handle import RunnerHandle

if t.TYPE_CHECKING:
    import tritonclient.grpc.aio as tritongrpcclient
    from tritonclient.grpc import service_pb2 as pb
    from fs.opener.registry import Registry

    P = t.ParamSpec("P")
    R = t.TypeVar("R")
else:
    tritongrpcclient - LazyLoader(
        "tritongrpcclient",
        globals(),
        "tritonclient.grpc.aio",
        exc_msg="tritonclient is required to use triton with BentoML. Install with 'pip install bentoml[triton]'.",
    )

logger = logging.getLogger(__name__)

__all__ = [
    "TritonRunner",
    "TritonRunnerHandle",
]

_object_setattr = object.__setattr__


class TritonRunner(Runner):
    name: str
    model_repository: str

    def __init__(
        self,
        name: str,
        model_repository: str,
        *,
        protocol: str | None = None,
    ):
        lname = name.lower()
        if name != lname:
            logger.warning(
                "Converting runner name '%s' to lowercase: '%s'", name, lname
            )

        # TODO: Support configuration (P1)

        # The logic below mimic the behaviour of Exportable.import_from
        # To determine models inside given model repository.
        try:
            parsed_model_repository = fs.opener.parse(model_repository)
        except fs.opener.errors.ParseError:
            if protocol is None:
                protocol = "osfs"
                resource: str = (
                    model_repository
                    if os.sep == "/"
                    else model_repository.replace(os.sep, "/")
                )
            else:
                resource: str = ""
        else:
            if protocol is not None:
                raise ValueError(
                    "An FS URL was passed as the output path; all additional information should be passed as part of the URL."
                )

            protocol = parsed_model_repository.protocol
            resource: str = parsed_model_repository.resource

        if protocol not in t.cast("Registry", fs.opener.registry).protocols:
            if protocol == "s3":
                raise ValueError(
                    "Tried to open an S3 url but the protocol is not registered; did you 'pip install \"bentoml[aws]\"'?"
                )
            else:
                raise ValueError(
                    f"Unknown or unsupported protocol {protocol}. Some supported protocols are 'ftp', 's3', and 'osfs'."
                )

        is_os_path = protocol in [
            "temp",
            "osfs",
            "userdata",
            "userconf",
            "sitedata",
            "siteconf",
            "usercache",
            "userlog",
        ]

        if is_os_path:
            try:
                repo_fs = fs.open_fs(f"{protocol}://{resource}")
            except fs.errors.CreateFailed:
                raise ValueError(f"Path {model_repository} does not exist.")
        else:
            resource = urllib.parse.quote(resource)
            repo_fs = fs.open_fs(f"{protocol}://{resource}")

        self.__attrs_init__(name=name, model_repository=model_repository)

        # List of models inside given model repository.
        for model in repo_fs.listdir("/"):
            _object_setattr(
                self,
                model,
                RunnerMethod(
                    self,
                    model,
                    # TODO: the following aren't relevant to TritonRunner yet.
                    RunnableMethodConfig(batchable=True, batch_dim=(0, 0)),
                    max_batch_size=0,
                    max_latency_ms=1000,
                ),
            )

    def init_local(self, quiet: bool = False) -> None:
        raise ValueError(
            "'init_local' is not supported for TritonRunner as this is intended to be used with Triton Inference Server."
        )

    def init_client(self):
        self._init(TritonRunnerHandle)


# NOTE: to support files, consider using ModelInferStream via raw_bytes_contents


@attr.define
class TritonRunnerHandle(RunnerHandle):
    runner: TritonRunner
    clean_context: contextlib.AsyncExitStack = attr.field(
        factory=contextlib.AsyncExitStack
    )
    # cache given client
    _client_cache: tritongrpcclient.InferenceServerClient = attr.field(
        init=False, default=None
    )
    # By default, the gRPC port from 'tritonserver' is 8001
    _address: str = attr.field(init=False, default="0.0.0.0:8001")

    async def is_ready(self, timeout: int) -> bool:
        return t.cast(bool, await self._client.is_server_live())

    @property
    def _client(self) -> tritongrpcclient.InferenceServerClient:
        from ...configuration import get_debug_mode

        if self._client_cache is None:
            try:
                self._client_cache = tritongrpcclient.InferenceServerClient(
                    url=self._address, verbose=get_debug_mode()
                )
            except Exception as e:
                import traceback

                logger.error(
                    "Failed to instantiate Triton Inference Server client for '%s', see details:",
                    self.runner.name,
                )
                logger.error(traceback.format_exc())
                raise e
        return self._client_cache

    def __del__(self):
        async def _():
            await self._client.close()

        asyncio.run(_())

    async def async_run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        from ...runner.container import AutoContainer

        # return metadata of a given model
        metadata: dict[
            str, pb.ModelMetadataResponse
        ] = await self._client.get_model_metadata(__bentoml_method.name)

        payload = Params[Payload](*args, **kwargs).map(
            functools.partial(AutoContainer.to_triton_payload, metadata=metadata)
        )

    def run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        ...
