from .utils.dataclasses import json_serializer as json_serializer
from multidict import CIMultiDict
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

BATCH_HEADER: str
HEADER_CHARSET: str
JSON_CHARSET: str
PathType: Any
MT = TypeVar('MT')

class FileLike:
    bytes_: Optional[bytes]
    uri: Optional[str]
    name: Optional[str]
    def __post_init__(self) -> None: ...
    @property
    def path(self): ...
    @property
    def stream(self): ...
    def read(self, size: int = ...): ...
    def seek(self, pos): ...
    def tell(self): ...
    def close(self) -> None: ...
    def __del__(self) -> None: ...

class HTTPHeaders(CIMultiDict):
    @property
    def content_type(self) -> str: ...
    @property
    def charset(self) -> Optional[str]: ...
    @property
    def content_encoding(self) -> str: ...
    @property
    def is_batch_input(self) -> Optional[bool]: ...
    @classmethod
    def from_dict(cls, d: Mapping[str, str]): ...
    @classmethod
    def from_sequence(cls, seq: Sequence[Tuple[str, str]]): ...
    def to_json(self): ...

class HTTPRequest:
    headers: HTTPHeaders
    body: bytes
    def __post_init__(self) -> None: ...
    @classmethod
    def parse_form_data(cls, self): ...
    @classmethod
    def from_flask_request(cls, request): ...
    def to_flask_request(self): ...

class HTTPResponse:
    status: int
    headers: HTTPHeaders
    body: Optional[bytes]
    @classmethod
    def new(cls, status: int = ..., headers: Union[HTTPHeaders, dict, tuple, list] = ..., body: bytes = ...): ...
    def __post_init__(self) -> None: ...
    def to_flask_response(self): ...
JsonSerializable = Union[bool, None, Dict, List, int, float, str]
AwsLambdaEvent = Union[Dict, List, str, int, float, None]
Input = TypeVar('Input')
Output = TypeVar('Output')
ApiFuncArgs = TypeVar('ApiFuncArgs')
BatchApiFuncArgs = TypeVar('BatchApiFuncArgs')
ApiFuncReturnValue = TypeVar('ApiFuncReturnValue')
BatchApiFuncReturnValue = TypeVar('BatchApiFuncReturnValue')

class InferenceResult:
    version: int
    data: Optional[Output]
    err_msg: str
    task_id: Optional[str]
    http_status: int
    http_headers: HTTPHeaders
    aws_lambda_event: Optional[dict]
    cli_status: Optional[int]
    def __post_init__(self) -> None: ...
    @classmethod
    def complete_discarded(cls, tasks: Iterable[InferenceTask], results: Iterable[InferenceResult]) -> Iterator[InferenceResult]: ...

class InferenceError(InferenceResult):
    http_status: int
    cli_status: int

class InferenceTask:
    version: int
    data: Optional[Input]
    error: Optional[InferenceResult]
    task_id: str
    is_discarded: bool
    batch: Optional[int]
    http_method: Optional[str]
    http_headers: HTTPHeaders
    aws_lambda_event: Optional[dict]
    cli_args: Optional[Sequence[str]]
    inference_job_args: Optional[Mapping[str, Any]]
    def discard(self, err_msg: str = ..., **context): ...
