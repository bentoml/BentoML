from __future__ import annotations

import typing as t
from types import TracebackType

import typing_extensions

from .. import InferInput as InferInput
from .. import InferResult as InferResult
from .. import raise_error as raise_error
from .. import service_pb2
from .. import get_error_grpc as get_error_grpc
from .. import KeepAliveOptions
from .. import raise_error_grpc as raise_error_grpc
from .. import ChannelCredentials
from .. import np_to_triton_dtype as np_to_triton_dtype
from .. import triton_to_np_dtype as triton_to_np_dtype
from .. import InferRequestedOutput as InferRequestedOutput
from .. import serialized_byte_size as serialized_byte_size
from .. import serialize_bf16_tensor as serialize_bf16_tensor
from .. import serialize_byte_tensor as serialize_byte_tensor
from .. import deserialize_bf16_tensor as deserialize_bf16_tensor
from .. import deserialize_bytes_tensor as deserialize_bytes_tensor
from .. import InferenceServerException

P = typing_extensions.ParamSpec("P")

class InferenceServerClient:
    """This feature is currently in beta and may be subject to change.

    An analogy of the tritonclient.grpc.InferenceServerClient to enable
    calling via asyncio syntax. The object is intended to be used by a single
    thread and simultaneously calling methods with different threads is not
    supported and can cause undefined behavior.

    """

    def __init__(
        self,
        url: str,
        verbose: bool = ...,
        ssl: bool = ...,
        root_certificates: str = ...,
        private_key: str = ...,
        certificate_chain: str = ...,
        creds: ChannelCredentials = ...,
        keepalive_options: KeepAliveOptions = ...,
        channel_args: list[tuple[str, t.Any]] = ...,
    ) -> None: ...
    async def __aenter__(self) -> InferenceServerClient: ...
    async def __aexit__(
        self, type: type[BaseException], value: Exception, traceback: TracebackType
    ) -> bool | None: ...
    async def close(self) -> None:
        """Close the client. Any future calls to server
        will result in an Error.

        """

    async def is_server_live(self, headers: dict[str, t.Any] = ...) -> bool:
        """Contact the inference server and get liveness.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Returns
        -------
        bool
            True if server is live, False if server is not live.

        Raises
        ------
        InferenceServerException
            If unable to get liveness.

        """

    async def is_server_ready(self, headers: dict[str, t.Any] = ...) -> bool:
        """Contact the inference server and get readiness.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Returns
        -------
        bool
            True if server is ready, False if server is not ready.

        Raises
        ------
        InferenceServerException
            If unable to get readiness.

        """

    async def is_model_ready(
        self,
        model_name: str,
        model_version: str = ...,
        headers: dict[str, t.Any] = ...,
    ) -> bool:
        """Contact the inference server and get the readiness of specified model.

        Parameters
        ----------
        model_name: str
            The name of the model to check for readiness.
        model_version: str
            The version of the model to check for readiness. The default value
            is an empty string which means then the server will choose a version
            based on the model and internal policy.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Returns
        -------
        bool
            True if the model is ready, False if not ready.

        Raises
        ------
        InferenceServerException
            If unable to get model readiness.

        """

    @t.overload
    async def get_server_metadata(
        self, headers: dict[str, t.Any] = ..., as_json: t.Literal[True] = ...
    ) -> dict[str, t.Any]: ...
    @t.overload
    async def get_server_metadata(
        self, headers: dict[str, t.Any] = ..., as_json: t.Literal[False] = ...
    ) -> service_pb2.ServerMetadataResponse:
        """Contact the inference server and get its metadata.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns server metadata as a json dict,
            otherwise as a protobuf message. Default value is
            False. The returned json is generated from the protobuf
            message using MessageToJson and as a result int64 values
            are represented as string. It is the caller's
            responsibility to convert these strings back to int64
            values as necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or ServerMetadataResponse message
            holding the metadata.

        Raises
        ------
        InferenceServerException
            If unable to get server metadata.

        """

    @t.overload
    async def get_model_metadata(
        self,
        model_name: str,
        model_version: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    async def get_model_metadata(
        self,
        model_name: str,
        model_version: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[False] = ...,
    ) -> service_pb2.ModelMetadataResponse:
        """Contact the inference server and get the metadata for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model
        model_version: str
            The version of the model to get metadata. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns model metadata as a json dict,
            otherwise as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or ModelMetadataResponse message holding
            the metadata.

        Raises
        ------
        InferenceServerException
            If unable to get model metadata.

        """

    @t.overload
    async def get_model_config(
        self,
        model_name: str,
        model_version: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    async def get_model_config(
        self,
        model_name: str,
        model_version: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[False] = ...,
    ) -> service_pb2.ModelConfigResponse:
        """Contact the inference server and get the configuration for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model
        model_version: str
            The version of the model to get configuration. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns configuration as a json dict, otherwise
            as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or ModelConfigResponse message holding
            the metadata.

        Raises
        ------
        InferenceServerException
            If unable to get model configuration.

        """

    @t.overload
    async def get_model_repository_index(
        self, headers: dict[str, t.Any] = ..., as_json: t.Literal[True] = ...
    ) -> dict[str, t.Any]: ...
    @t.overload
    async def get_model_repository_index(
        self, headers: dict[str, t.Any] = ..., as_json: t.Literal[False] = ...
    ) -> service_pb2.RepositoryIndexResponse:
        """Get the index of model repository contents

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns model repository index
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or RepositoryIndexResponse message holding
            the model repository index.

        """
        ...

    async def load_model(
        self,
        model_name: str,
        headers: dict[str, t.Any] = ...,
        config: str = ...,
        files: dict[str, str] = ...,
    ) -> None:
        """Request the inference server to load or reload specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be loaded.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        config: str
            Optional JSON representation of a model config provided for
            the load request, if provided, this config will be used for
            loading the model.
        files: dict
            Optional dictionary specifying file path (with "file:" prefix) in
            the override model directory to the file content as bytes.
            The files will form the model directory that the model will be
            loaded from. If specified, 'config' must be provided to be
            the model configuration of the override model directory.

        Raises
        ------
        InferenceServerException
            If unable to load the model.

        """

    async def unload_model(
        self,
        model_name: str,
        headers: dict[str, t.Any] = ...,
        unload_dependents: bool = ...,
    ) -> None:
        """Request the inference server to unload specified model.

        Parameters
        ----------
        model_name: str : str
            The name of the model to be unloaded.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        unload_dependents : bool
            Whether the dependents of the model should also be unloaded.

        Raises
        ------
        InferenceServerException
            If unable to unload the model.

        """

    @t.overload
    async def get_inference_statistics(
        self,
        model_name: str = ...,
        model_version: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    async def get_inference_statistics(
        self,
        model_name: str = ...,
        model_version: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[False] = ...,
    ) -> service_pb2.ModelStatisticsResponse:
        """Get the inference statistics for the specified model name and
        version.

        Parameters
        ----------
        model_name : str
            The name of the model to get statistics. The default value is
            an empty string, which means statistics of all models will
            be returned.
        model_version: str
            The version of the model to get inference statistics. The
            default value is an empty string which means then the server
            will return the statistics of all available model versions.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns inference statistics
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Raises
        ------
        InferenceServerException
            If unable to get the model inference statistics.

        """

    @t.overload
    async def update_trace_settings(
        self,
        model_name: str = ...,
        settings: dict[str, t.Any] = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    async def update_trace_settings(
        self,
        model_name: str = ...,
        settings: dict[str, t.Any] = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[False] = ...,
    ) -> service_pb2.TraceSettingResponse:
        """Update the trace settings for the specified model name, or
        global trace settings if model name is not given.
        Returns the trace settings after the update.

        Parameters
        ----------
        model_name : str
            The name of the model to update trace settings. Specifying None or
            empty string will update the global trace settings.
            The default value is None.
        settings: dict
            The new trace setting values. Only the settings listed will be
            updated. If a trace setting is listed in the dictionary with
            a value of 'None', that setting will be cleared.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns trace settings
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or TraceSettingResponse message holding
            the updated trace settings.

        Raises
        ------
        InferenceServerException
            If unable to update the trace settings.

        """
        ...

    @t.overload
    async def get_trace_settings(
        self,
        model_name: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    async def get_trace_settings(
        self,
        model_name: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[False] = ...,
    ) -> service_pb2.TraceSettingResponse:
        """Get the trace settings for the specified model name, or global trace
        settings if model name is not given

        Parameters
        ----------
        model_name : str
            The name of the model to get trace settings. Specifying None or
            empty string will return the global trace settings.
            The default value is None.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns trace settings
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or TraceSettingResponse message holding
            the trace settings.

        Raises
        ------
        InferenceServerException
            If unable to get the trace settings.

        """

    @t.overload
    async def update_log_settings(
        self,
        settings: dict[str, t.Any],
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    async def update_log_settings(
        self,
        settings: dict[str, t.Any],
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[False] = ...,
    ) -> service_pb2.LogSettingsResponse:
        """Update the global log settings.
        Returns the log settings after the update.
        Parameters
        ----------
        settings: dict
            The new log setting values. Only the settings listed will be
            updated.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns trace settings
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.
        Returns
        -------
        dict or protobuf message
            The JSON dict or LogSettingsResponse message holding
            the updated log settings.
        Raises
        ------
        InferenceServerException
            If unable to update the log settings.
        """

    @t.overload
    async def get_log_settings(
        self, headers: dict[str, t.Any] = ..., as_json: t.Literal[True] = ...
    ) -> dict[str, t.Any]: ...
    @t.overload
    async def get_log_settings(
        self, headers: dict[str, t.Any] = ..., as_json: t.Literal[False] = ...
    ) -> service_pb2.LogSettingsResponse:
        """Get the global log settings.
        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns log settings
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.
        Returns
        -------
        dict or protobuf message
            The JSON dict or LogSettingsResponse message holding
            the log settings.
        Raises
        ------
        InferenceServerException
            If unable to get the log settings.
        """

    @t.overload
    async def get_system_shared_memory_status(
        self,
        region_name: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    async def get_system_shared_memory_status(
        self,
        region_name: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[False] = ...,
    ) -> service_pb2.SystemSharedMemoryStatusResponse:
        """Request system shared memory status from the server.

        Parameters
        ----------
        region_name : str
            The name of the region to query status. The default
            value is an empty string, which means that the status
            of all active system shared memory will be returned.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns system shared memory status as a json
            dict, otherwise as a protobuf message. Default value is
            False.  The returned json is generated from the protobuf
            message using MessageToJson and as a result int64 values
            are represented as string. It is the caller's
            responsibility to convert these strings back to int64
            values as necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or SystemSharedMemoryStatusResponse message holding
            the system shared memory status.

        Raises
        ------
        InferenceServerException
            If unable to get the status of specified shared memory.

        """

    async def register_system_shared_memory(
        self,
        name: str,
        key: str,
        byte_size: int,
        offset: int = ...,
        headers: dict[str, t.Any] = ...,
    ) -> None:
        """Request the server to register a system shared memory with the
        following specification.

        Parameters
        ----------
        name : str
            The name of the region to register.
        key : str
            The key of the underlying memory object that contains the
            system shared memory region.
        byte_size : int
            The size of the system shared memory region, in bytes.
        offset : int
            Offset, in bytes, within the underlying memory object to
            the start of the system shared memory region. The default
            value is zero.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to register the specified system shared memory.

        """

    async def unregister_system_shared_memory(
        self, name: str = ..., headers: dict[str, t.Any] = ...
    ) -> None:
        """Request the server to unregister a system shared memory with the
        specified name.

        Parameters
        ----------
        name : str
            The name of the region to unregister. The default value is empty
            string which means all the system shared memory regions will be
            unregistered.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to unregister the specified system shared memory region.

        """

    @t.overload
    async def get_cuda_shared_memory_status(
        self,
        region_name: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    async def get_cuda_shared_memory_status(
        self,
        region_name: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[False] = ...,
    ) -> service_pb2.CudaSharedMemoryStatusResponse:
        """Request cuda shared memory status from the server.

        Parameters
        ----------
        region_name : str
            The name of the region to query status. The default
            value is an empty string, which means that the status
            of all active cuda shared memory will be returned.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns cuda shared memory status as a json
            dict, otherwise as a protobuf message. Default value is
            False.  The returned json is generated from the protobuf
            message using MessageToJson and as a result int64 values
            are represented as string. It is the caller's
            responsibility to convert these strings back to int64
            values as necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or CudaSharedMemoryStatusResponse message holding
            the cuda shared memory status.

        Raises
        ------
        InferenceServerException
            If unable to get the status of specified shared memory.

        """

    async def register_cuda_shared_memory(
        self,
        name: str,
        raw_handle: bytes,
        device_id: int,
        byte_size: int,
        headers: dict[str, t.Any] = ...,
    ) -> None:
        """Request the server to register a system shared memory with the
        following specification.

        Parameters
        ----------
        name : str
            The name of the region to register.
        raw_handle : bytes
            The raw serialized cudaIPC handle in base64 encoding.
        device_id : int
            The GPU device ID on which the cudaIPC handle was created.
        byte_size : int
            The size of the cuda shared memory region, in bytes.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to register the specified cuda shared memory.

        """

    async def unregister_cuda_shared_memory(
        self, name: str = ..., headers: dict[str, t.Any] = ...
    ) -> None:
        """Request the server to unregister a cuda shared memory with the
        specified name.

        Parameters
        ----------
        name : str
            The name of the region to unregister. The default value is empty
            string which means all the cuda shared memory regions will be
            unregistered.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to unregister the specified cuda shared memory region.

        """

    async def infer(
        self,
        model_name: str,
        inputs: list[InferInput],
        model_version: str = ...,
        outputs: list[InferRequestedOutput] = ...,
        request_id: str = ...,
        sequence_id: str = ...,
        sequence_start: bool = ...,
        sequence_end: bool = ...,
        priority: int = ...,
        timeout: int = ...,
        client_timeout: int = ...,
        headers: dict[str, t.Any] = ...,
        compression_algorithm: str = ...,
    ) -> InferResult:
        """Run synchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'.

        Parameters
        ----------
        model_name: str
            The name of the model to run inference.
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        model_version : str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        outputs : list
            A list of InferRequestedOutput objects, each describing how the output
            data must be returned. If not specified all outputs produced
            by the model will be returned using default settings.
        request_id : str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is an empty string which means no
            request_id will be used.
        sequence_id : int
            The unique identifier for the sequence being represented by the
            object. Default value is 0 which means that the request does not
            belong to a sequence.
        sequence_start : bool
            Indicates whether the request being added marks the start of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        sequence_end : bool
            Indicates whether the request being added marks the end of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        priority : int
            Indicates the priority of the request. Priority value zero
            indicates that the default priority level should be used
            (i.e. same behavior as not specifying the priority parameter).
            Lower value priorities indicate higher priority levels. Thus
            the highest priority level is indicated by setting the parameter
            to 1, the next highest is 2, etc. If not provided, the server
            will handle the request using default setting for the model.
        timeout : int
            The timeout value for the request, in microseconds. If the request
            cannot be completed within the time the server can take a
            model-specific action such as terminating the request. If not
            provided, the server will handle the request using default setting
            for the model.
        client_timeout : float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.
        headers : dict
            Optional dictionary specifying additional HTTP headers to include
            in the request.
        compression_algorithm : str
            Optional grpc compression algorithm to be used on client side.
            Currently supports "deflate", "gzip" and None. By default, no
            compression is used.

        Returns
        -------
        InferResult
            The object holding the result of the inference.

        Raises
        ------
        InferenceServerException
            If server fails to perform inference.
        """

    async def stream_infer(
        self,
        inputs_iterator: t.AsyncGenerator[
            dict[
                str,
                t.Concatenate[
                    str,
                    list[InferInput],
                    str,
                    list[InferRequestedOutput],
                    str,
                    str | int,
                    bool,
                    bool,
                    int,
                    int,
                    P,
                ],
            ],
            None,
        ],
        stream_timeout: float = ...,
        headers: dict[str, t.Any] = ...,
        compression_algorithm: str = ...,
    ) -> t.Generator[
        tuple[InferResult | None, InferenceServerException | None], None, None
    ]:
        """Runs an asynchronous inference over gRPC bi-directional streaming
        API.

        Parameters
        ----------
        inputs_iterator : async_generator
            Async iterator that yields a dict(s) consists of the input
            parameters to the async_stream_infer function defined in
            tritonclient.grpc.InferenceServerClient.
        stream_timeout : float
            Optional stream timeout. The stream will be closed once the
            specified timeout expires.
        headers: dict
            Optional dictionary specifying additional HTTP headers to include
            in the request.
        compression_algorithm : str
            Optional grpc compression algorithm to be used on client side.
            Currently supports "deflate", "gzip" and None. By default, no
            compression is used.

        Returns
        -------
        async_generator
            Yield tuple holding (InferResult, InferenceServerException) objects.

        Raises
        ------
        InferenceServerException
            If inputs_iterator does not yield the correct input.

        """
