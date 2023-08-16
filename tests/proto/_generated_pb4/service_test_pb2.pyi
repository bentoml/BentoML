"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import abc
import builtins
import collections.abc
import concurrent.futures
import google.protobuf.descriptor
import google.protobuf.message
import google.protobuf.service
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class ExecuteRequest(google.protobuf.message.Message):
    """Represents a request for TestService."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    INPUT_FIELD_NUMBER: builtins.int
    input: builtins.str
    def __init__(
        self,
        *,
        input: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["input", b"input"]) -> None: ...

global___ExecuteRequest = ExecuteRequest

@typing_extensions.final
class ExecuteResponse(google.protobuf.message.Message):
    """Represents a response from TestService."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    OUTPUT_FIELD_NUMBER: builtins.int
    output: builtins.str
    def __init__(
        self,
        *,
        output: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["output", b"output"]) -> None: ...

global___ExecuteResponse = ExecuteResponse

class TestService(google.protobuf.service.Service, metaclass=abc.ABCMeta):
    """Use for testing interceptors per RPC call."""

    DESCRIPTOR: google.protobuf.descriptor.ServiceDescriptor
    @abc.abstractmethod
    def Execute(
        inst: TestService,  # pyright: ignore[reportSelfClsParameterName]
        rpc_controller: google.protobuf.service.RpcController,
        request: global___ExecuteRequest,
        callback: collections.abc.Callable[[global___ExecuteResponse], None] | None,
    ) -> concurrent.futures.Future[global___ExecuteResponse]:
        """Unary API"""

class TestService_Stub(TestService):
    """Use for testing interceptors per RPC call."""

    def __init__(self, rpc_channel: google.protobuf.service.RpcChannel) -> None: ...
    DESCRIPTOR: google.protobuf.descriptor.ServiceDescriptor
    def Execute(
        inst: TestService_Stub,  # pyright: ignore[reportSelfClsParameterName]
        rpc_controller: google.protobuf.service.RpcController,
        request: global___ExecuteRequest,
        callback: collections.abc.Callable[[global___ExecuteResponse], None] | None = ...,
    ) -> concurrent.futures.Future[global___ExecuteResponse]:
        """Unary API"""
