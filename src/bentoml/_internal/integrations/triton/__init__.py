from __future__ import annotations

import os
import typing as t
import logging

import attr
import urllib.parse
import fs
import fs.errors
import fs.opener
import fs.opener.errors

from ...utils import LazyLoader
from ...runner.runnable import RunnableMethodConfig
from ...runner.runner import Runner, RunnerMethod
from ...runner.runner_handle import DummyRunnerHandle, RunnerHandle

if t.TYPE_CHECKING:
    from fs.opener.registry import Registry
    from google.protobuf import text_format

    import tritonclient.grpc as tritongrpcclient

    P = t.ParamSpec("P")
    R = t.TypeVar("R")
else:
    pb_model_config = LazyLoader(
        "pb_model_config",
        globals(),
        "bentoml._internal.integrations.triton.model_config_pb2",
    )
    text_format = LazyLoader(
        "text_format",
        globals(),
        "google.protobuf.text_format",
        exc_msg="'protobuf' is required to use triton with BentoML. Install with 'pip install bentoml[triton]'.",
    )
    tritongrpcclient - LazyLoader(
        "tritongrpcclient",
        globals(),
        "tritonclient.grpc",
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

        model_names = repo_fs.listdir("/")

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
                    RunnableMethodConfig(batchable=False, batch_dim=(0, 0)),
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


@attr.define
class TritonRunnerHandle(RunnerHandle):
    runner: TritonRunner

    async def is_ready(self, timeout: int) -> bool:
        return True

    @property
    def _client(self) -> tritongrpcclient.InferenceServerClient:
        from ...configuration import get_debug_mode

        try:
            return tritongrpcclient.InferenceServerClient(
                url="0.0.0.0:8001", verbose=get_debug_mode()
            )
        except Exception as e:
            import traceback

            logger.error(
                "Failed to instantiate Triton Inference Server client for '%s', see details:",
                self.runner.name,
            )
            logger.error(traceback.format_exc())
            raise e

    def run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        ...

    async def async_run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        ...
