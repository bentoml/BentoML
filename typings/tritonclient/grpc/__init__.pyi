from __future__ import annotations

import typing as t
from types import TracebackType

from numpy.typing import NDArray

from . import service_pb2
from . import model_config_pb2 as model_config_pb2
from . import service_pb2_grpc as service_pb2_grpc
from ..utils import raise_error as raise_error
from ..utils import np_to_triton_dtype as np_to_triton_dtype
from ..utils import triton_to_np_dtype as triton_to_np_dtype
from ..utils import serialized_byte_size as serialized_byte_size
from ..utils import serialize_bf16_tensor as serialize_bf16_tensor
from ..utils import serialize_byte_tensor as serialize_byte_tensor
from ..utils import deserialize_bf16_tensor as deserialize_bf16_tensor
from ..utils import deserialize_bytes_tensor as deserialize_bytes_tensor
from ..utils import InferenceServerException

INT32_MAX: int = ...
MAX_GRPC_MESSAGE_SIZE = INT32_MAX

ChannelCredentials = t.Any

class RpcError(Exception):
    """Exception raised by gRPC"""

def get_error_grpc(rpc_error: RpcError) -> InferenceServerException: ...
def raise_error_grpc(rpc_error: RpcError) -> t.NoReturn: ...

class KeepAliveOptions:
    """A KeepAliveOptions object is used to encapsulate GRPC KeepAlive
    related parameters for initiating an InferenceServerclient object.

    See the https://github.com/grpc/grpc/blob/master/doc/keepalive.md
    documentation for more information.

    Parameters
    ----------
    keepalive_time_ms: int
        The period (in milliseconds) after which a keepalive ping is sent on
        the transport. Default is INT32_MAX.

    keepalive_timeout_ms: int
        The period (in milliseconds) the sender of the keepalive ping waits
        for an acknowledgement. If it does not receive an acknowledgment
        within this time, it will close the connection. Default is 20000
        (20 seconds).

    keepalive_permit_without_calls: bool
        Allows keepalive pings to be sent even if there are no calls in flight.
        Default is False.

    http2_max_pings_without_data: int
        The maximum number of pings that can be sent when there is no
        data/header frame to be sent. gRPC Core will not continue sending
        pings if we run over the limit. Setting it to 0 allows sending pings
        without such a restriction. Default is 2.

    """

    def __init__(
        self,
        keepalive_time_ms: int = ...,
        keepalive_timeout_ms: int = ...,
        keepalive_permit_without_calls: bool = ...,
        http2_max_pings_without_data: int = ...,
    ) -> None: ...

class InferenceServerClient:
    """An InferenceServerClient object is used to perform any kind of
    communication with the InferenceServer using gRPC protocol. Most
    of the methods are thread-safe except start_stream, stop_stream
    and async_stream_infer. Accessing a client stream with different
    threads will cause undefined behavior.

    Parameters
    ----------
    url : str
        The inference server URL, e.g. 'localhost:8001'.

    verbose : bool
        If True generate verbose output. Default value is False.

    ssl : bool
        If True use SSL encrypted secure channel. Default is False.

    root_certificates : str
        File holding the PEM-encoded root certificates as a byte
        string, or None to retrieve them from a default location
        chosen by gRPC runtime. The option is ignored if `ssl`
        is False. Default is None.

    private_key : str
        File holding the PEM-encoded private key as a byte string,
        or None if no private key should be used. The option is
        ignored if `ssl` is False. Default is None.

    certificate_chain : str
        File holding PEM-encoded certificate chain as a byte string
        to use or None if no certificate chain should be used. The
        option is ignored if `ssl` is False. Default is None.

    creds: grpc.ChannelCredentials
        A grpc.ChannelCredentials object to use for the connection.
        The ssl, root_certificates, private_key and certificate_chain
        options will be ignored when using this option. Default is None.

    keepalive_options: KeepAliveOptions
        Object encapsulating various GRPC KeepAlive options. See
        the class definition for more information. Default is None.

    channel_args: List[Tuple]
        List of Tuple pairs ("key", value) to be passed directly to the GRPC
        channel as the channel_arguments. If this argument is provided, it is
        expected the channel arguments are correct and complete, and the
        keepalive_options parameter will be ignored since the corresponding
        keepalive channel arguments can be set directly in this parameter. See
        https://grpc.github.io/grpc/python/glossary.html#term-channel_arguments
        for more details. Default is None.

    Raises
    ------
    Exception
        If unable to create a client.

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
    def __enter__(self) -> InferenceServerClient: ...
    def __exit__(
        self, type: type[BaseException], value: BaseException, traceback: TracebackType
    ) -> bool | None: ...
    def __del__(self) -> None: ...
    def close(self) -> None:
        """Close the client. Any future calls to server
        will result in an Error.

        """

    def is_server_live(self, headers: dict[str, t.Any] = ...) -> bool:
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

    def is_server_ready(self, headers: dict[str, t.Any] = ...) -> bool:
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

    def is_model_ready(
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
    def get_server_metadata(
        self, headers: dict[str, t.Any] = ..., as_json: t.Literal[True] = ...
    ) -> dict[str, t.Any]: ...
    @t.overload
    def get_server_metadata(
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
    def get_model_metadata(
        self,
        model_name: str,
        model_version: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    def get_model_metadata(
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
    def get_model_config(
        self,
        model_name: str,
        model_version: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    def get_model_config(
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
    def get_model_repository_index(
        self, headers: dict[str, t.Any] = ..., as_json: t.Literal[True] = ...
    ) -> dict[str, t.Any]: ...
    @t.overload
    def get_model_repository_index(
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

    def load_model(
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

    def unload_model(
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
    def get_inference_statistics(
        self,
        model_name: str = ...,
        model_version: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    def get_inference_statistics(
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
    def update_trace_settings(
        self,
        model_name: str = ...,
        settings: dict[str, t.Any] = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    def update_trace_settings(
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
    def get_trace_settings(
        self,
        model_name: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    def get_trace_settings(
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
    def update_log_settings(
        self,
        settings: dict[str, t.Any],
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    def update_log_settings(
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
    def get_log_settings(
        self, headers: dict[str, t.Any] = ..., as_json: t.Literal[True] = ...
    ) -> dict[str, t.Any]: ...
    @t.overload
    def get_log_settings(
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
    def get_system_shared_memory_status(
        self,
        region_name: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    def get_system_shared_memory_status(
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

    def register_system_shared_memory(
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

    def unregister_system_shared_memory(
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
    def get_cuda_shared_memory_status(
        self,
        region_name: str = ...,
        headers: dict[str, t.Any] = ...,
        as_json: t.Literal[True] = ...,
    ) -> dict[str, t.Any]: ...
    @t.overload
    def get_cuda_shared_memory_status(
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

    def register_cuda_shared_memory(
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

    def unregister_cuda_shared_memory(
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

    def infer(
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

    def async_infer(
        self,
        model_name: str,
        inputs: list[InferInput],
        callback: t.Callable[[InferResult, InferenceServerException], None],
        model_version: str = ...,
        outputs: list[InferRequestedOutput] = ...,
        request_id: str = ...,
        sequence_id: int = ...,
        sequence_start: bool = ...,
        sequence_end: bool = ...,
        priority: int = ...,
        timeout: int = ...,
        client_timeout: int = ...,
        headers: dict[str, t.Any] = ...,
        compression_algorithm: str = ...,
    ) -> None:
        """Run asynchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'.

        Parameters
        ----------
        model_name: str
            The name of the model to run inference.
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        callback : function
            Python function that is invoked once the request is completed.
            The function must reserve the last two arguments (result, error)
            to hold InferResult and InferenceServerException objects
            respectively which will be provided to the function when executing
            the callback. The ownership of these objects will be given to the
            user. The 'error' would be None for a successful inference.
        model_version: str
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
        sequence_start: bool
            Indicates whether the request being added marks the start of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        sequence_end: bool
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
            to take. The client will abort request and provide
            error with message "Deadline Exceeded" in the callback when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        compression_algorithm : str
            Optional grpc compression algorithm to be used on client side.
            Currently supports "deflate", "gzip" and None. By default, no
            compression is used.

        Raises
        ------
        InferenceServerException
            If server fails to issue inference.
        """

    def start_stream(
        self,
        callback: t.Callable[[InferResult, InferenceServerException], None],
        stream_timeout: float = ...,
        headers: dict[str, t.Any] = ...,
        compression_algorithm: str = ...,
    ) -> None:
        """Starts a grpc bi-directional stream to send streaming inferences.
        Note: When using stream, user must ensure the InferenceServerClient.close()
        gets called at exit.

        Parameters
        ----------
        callback : function
            Python function that is invoked upon receiving response from
            the underlying stream. The function must reserve the last two
            arguments (result, error) to hold InferResult and
            InferenceServerException objects respectively which will be
            provided to the function when executing the callback. The
            ownership of these objects will be given to the user. The
            'error' would be None for a successful inference.
        stream_timeout : float
            Optional stream timeout. The stream will be closed once the
            specified timeout expires.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        compression_algorithm : str
            Optional grpc compression algorithm to be used on client side.
            Currently supports "deflate", "gzip" and None. By default, no
            compression is used.

        Raises
        ------
        InferenceServerException
            If unable to start a stream or a stream was already running
            for this client.

        """

    def stop_stream(self) -> None:
        """Stops a stream if one available."""

    def async_stream_infer(
        self,
        model_name: str,
        inputs: list[InferInput],
        model_version: str = ...,
        outputs: list[InferRequestedOutput] = ...,
        request_id: str = ...,
        sequence_id: str | int = ...,
        sequence_start: bool = ...,
        sequence_end: bool = ...,
        priority: int = ...,
        timeout: int = ...,
    ) -> None:
        """Runs an asynchronous inference over gRPC bi-directional streaming
        API. A stream must be established with a call to start_stream()
        before calling this function. All the results will be provided to the
        callback function associated with the stream.

        Parameters
        ----------
        model_name: str
            The name of the model to run inference.
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        model_version: str
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
        sequence_id : int or str
            The unique identifier for the sequence being represented by the
            object.  A value of 0 or "" means that the request does not
            belong to a sequence. Default is 0.
        sequence_start: bool
            Indicates whether the request being added marks the start of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0 or "".
        sequence_end: bool
            Indicates whether the request being added marks the end of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0 or "".
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

        Raises
        ------
        InferenceServerException
            If server fails to issue inference.
        """

class InferInput:
    """An object of InferInput class is used to describe
    input tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of input whose data will be described by this object
    shape : list
        The shape of the associated input.
    datatype : str
        The datatype of the associated input.

    """

    def __init__(self, name: str, shape: tuple[int, ...], datatype: str) -> None: ...
    def name(self) -> str:
        """Get the name of input associated with this object.

        Returns
        -------
        str
            The name of input
        """

    def datatype(self) -> str:
        """Get the datatype of input associated with this object.

        Returns
        -------
        str
            The datatype of input
        """

    def shape(self) -> list[int]:
        """Get the shape of input associated with this object.

        Returns
        -------
        list
            The shape of input
        """

    def set_shape(self, shape: list[int]) -> None:
        """Set the shape of input.

        Parameters
        ----------
        shape : list
            The shape of the associated input.
        """

    def set_data_from_numpy(self, input_tensor: NDArray[t.Any]) -> None:
        """Set the tensor data from the specified numpy array for
        input associated with this object.

        Parameters
        ----------
        input_tensor : numpy array
            The tensor data in numpy array format

        Raises
        ------
        InferenceServerException
            If failed to set data for the tensor.
        """

    def set_shared_memory(
        self, region_name: str, byte_size: int, offset: int = ...
    ) -> None:
        """Set the tensor data from the specified shared memory region.

        Parameters
        ----------
        region_name : str
            The name of the shared memory region holding tensor data.
        byte_size : int
            The size of the shared memory region holding tensor data.
        offset : int
            The offset, in bytes, into the region where the data for
            the tensor starts. The default value is 0.

        """

class InferRequestedOutput:
    """An object of InferRequestedOutput class is used to describe a
    requested output tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of output tensor to associate with this object
    class_count : int
        The number of classifications to be requested. The default
        value is 0 which means the classification results are not
        requested.
    """

    def __init__(self, name: str, class_count: int = ...) -> None: ...
    def name(self) -> str:
        """Get the name of output associated with this object.

        Returns
        -------
        str
            The name of output
        """

    def set_shared_memory(
        self, region_name: str, byte_size: int, offset: int = ...
    ) -> None:
        """Marks the output to return the inference result in
        specified shared memory region.

        Parameters
        ----------
        region_name : str
            The name of the shared memory region to hold tensor data.
        byte_size : int
            The size of the shared memory region to hold tensor data.
        offset : int
            The offset, in bytes, into the region where the data for
            the tensor starts. The default value is 0.

        Raises
        ------
        InferenceServerException
            If failed to set shared memory for the tensor.
        """

    def unset_shared_memory(self) -> None:
        """Clears the shared memory option set by the last call to
        InferRequestedOutput.set_shared_memory(). After call to this
        function requested output will no longer be returned in a
        shared memory region.
        """

class InferResult:
    """An object of InferResult class holds the response of
    an inference request and provide methods to retrieve
    inference results.

    Parameters
    ----------
    result : protobuf message
        The ModelInferResponse returned by the server
    """

    def __init__(self, result: service_pb2.ModelInferResponse) -> None: ...
    def as_numpy(self, name: str) -> NDArray[t.Any]:
        """Get the tensor data for output associated with this object
        in numpy format

        Parameters
        ----------
        name : str
            The name of the output tensor whose result is to be retrieved.

        Returns
        -------
        numpy array
            The numpy array containing the response data for the tensor or
            None if the data for specified tensor name is not found.
        """

    @t.overload
    def get_output(
        self, name: str, as_json: t.Literal[True] = ...
    ) -> dict[str, t.Any]: ...
    @t.overload
    def get_output(
        self, name: str, as_json: t.Literal[False] = ...
    ) -> service_pb2.ModelInferResponse.InferOutputTensor:
        """Retrieves the InferOutputTensor corresponding to the
        named ouput.

        Parameters
        ----------
        name : str
            The name of the tensor for which Output is to be
            retrieved.
        as_json : bool
            If True then returns response as a json dict, otherwise
            as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        protobuf message or dict
            If a InferOutputTensor with specified name is present in
            ModelInferResponse then returns it as a protobuf messsage
            or dict, otherwise returns None.
        """

    @t.overload
    def get_response(self, as_json: t.Literal[True] = ...) -> dict[str, t.Any]: ...
    @t.overload
    def get_response(
        self, as_json: t.Literal[False] = ...
    ) -> service_pb2.ModelInferResponse:
        """Retrieves the complete ModelInferResponse as a
        json dict object or protobuf message

        Parameters
        ----------
        as_json : bool
            If True then returns response as a json dict, otherwise
            as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        protobuf message or dict
            The underlying ModelInferResponse as a protobuf message or dict.
        """
