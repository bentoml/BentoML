from __future__ import annotations

import os
import time
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

from ...utils import LazyLoader
from ...utils import resolve_user_filepath
from ...runner.utils import Params
from ...runner.runner import Runner
from ...runner.runner import RunnerMethod
from ...runner.runnable import RunnableMethodConfig
from ...runner.runner_handle import RunnerHandle

if t.TYPE_CHECKING:
    import tritonclient.grpc.aio as tritongrpcclient
    from fs.base import FS
    from fs.opener.registry import Registry

    from ... import external_typing as ext

    P = t.ParamSpec("P")
    R = t.TypeVar("R")
else:
    tritongrpcclient = LazyLoader(
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
    _fs: FS
    repository_path: str

    def __init__(
        self,
        name: str,
        model_repository: str,
        *,
        fs_protocol: str | None = None,
    ):
        lname = name.lower()
        if name != lname:
            logger.warning(
                "Converting runner name '%s' to lowercase: '%s'", name, lname
            )

        _object_setattr(self, "name", lname)
        _object_setattr(
            self, "repository_path", resolve_user_filepath(model_repository, ctx=None)
        )

        # TODO: Support configuration (P1)

        # The logic below mimic the behaviour of Exportable.import_from
        # To determine models inside given model repository.
        try:
            parsed_model_repository = fs.opener.parse(model_repository)
        except fs.opener.errors.ParseError:
            if fs_protocol is None:
                fs_protocol = "osfs"
                resource: str = (
                    model_repository
                    if os.sep == "/"
                    else model_repository.replace(os.sep, "/")
                )
            else:
                resource: str = ""
        else:
            if fs_protocol is not None:
                raise ValueError(
                    "An FS URL was passed as the output path; all additional information should be passed as part of the URL."
                )

            fs_protocol = parsed_model_repository.protocol
            resource: str = parsed_model_repository.resource

        if fs_protocol not in t.cast("Registry", fs.opener.registry).protocols:
            if fs_protocol == "s3":
                raise ValueError(
                    "Tried to open an S3 url but the protocol is not registered; did you 'pip install \"bentoml[aws]\"'?"
                )
            else:
                raise ValueError(
                    f"Unknown or unsupported protocol {fs_protocol}. Some supported protocols are 'ftp', 's3', and 'osfs'."
                )

        is_os_path = fs_protocol in [
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
                repo_fs = fs.open_fs(f"{fs_protocol}://{resource}")
            except fs.errors.CreateFailed:
                raise ValueError(f"Path {model_repository} does not exist.")
        else:
            resource = urllib.parse.quote(resource)
            repo_fs = fs.open_fs(f"{fs_protocol}://{resource}")

        _object_setattr(self, "_fs", repo_fs)

        # List of models inside given model repository.
        for model in self._fs.listdir("/"):
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
        start = time.time()
        while time.time() - start < timeout:
            try:
                if (
                    await self._client.is_server_ready()
                    and await self._client.is_server_live()
                ):
                    return True
                else:
                    await asyncio.sleep(1)
            except Exception as err:
                logger.error(
                    "Caught exception while waiting Triton to be ready: %s", err
                )
                await asyncio.sleep(1)
        return False

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
        __bentoml_method: RunnerMethod[t.Any, P, t.Any],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tritongrpcclient.InferResult:
        from ...runner.container import AutoContainer

        assert (
            len(args) == 0 ^ len(kwargs) == 0
        ), f"Inputs for model '{__bentoml_method.name}' can be given either as positional (args) or keyword arguments (kwargs), but not both. \
                See https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#model-configuration"

        model_name = __bentoml_method.name

        # return metadata of a given model
        model_metadata = await self._client.get_model_metadata(model_name)

        pass_args = args if len(args) > 0 else kwargs
        if len(model_metadata.inputs) != len(pass_args):
            raise ValueError(
                f"Number of provided arguments ({len(model_metadata.inputs)}) does not match the number of inputs ({len(pass_args)})"
            )

        input_params = Params[ext.NpNDArray](*args, **kwargs).map(
            AutoContainer.to_triton_payload
        )

        outputs = [
            tritongrpcclient.InferRequestedOutput(output.name)
            for output in model_metadata.outputs
        ]
        inputs: list[tritongrpcclient.InferInput] = []

        if len(args) > 0:
            for (infer_input, arg) in zip(model_metadata.inputs, input_params.args):
                InferInput = tritongrpcclient.InferInput(
                    infer_input.name,
                    arg.shape,
                    tritongrpcclient.np_to_triton_dtype(arg.dtype),
                )
                InferInput.set_data_from_numpy(arg)
                inputs.append(InferInput)
        else:
            for infer_input in model_metadata.inputs:
                arg = input_params.kwargs[infer_input.name]
                inputs.append(
                    tritongrpcclient.InferInput(
                        infer_input.name,
                        arg.shape,
                        tritongrpcclient.np_to_triton_dtype(arg.dtype),
                    )
                )
                inputs[-1].set_data_from_numpy(arg)

        try:
            return await self._client.infer(
                model_name=model_name, inputs=inputs, outputs=outputs
            )
        except tritongrpcclient.InferenceServerException as err:
            logger.error("Caught exception while sending payload to Triton:")
            logger.error(err)
            raise err

    def run_method(
        self,
        __bentoml_method: RunnerMethod[t.Any, P, t.Any],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tritongrpcclient.InferResult:
        import anyio

        return anyio.from_thread.run(
            functools.partial(self.async_run_method, **kwargs),
            __bentoml_method,
            *args,
        )
