import typing as t
import contextvars
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import starlette.requests
    import starlette.responses


class Metadata(ABC):
    @abstractmethod
    def __setitem__(self, key: str, value: str) -> None:
        """
        Set the header ``key`` to ``value``, removing any duplicate entries.
        Retains insertion order.
        """

    @abstractmethod
    def __delitem__(self, key: str) -> None:
        """
        Remove the header ``key``.
        """

    @abstractmethod
    def __ior__(self, other: t.Mapping[t.Any, t.Any]) -> "Metadata":
        """
        Updates this metadata with the contents of ``other``
        """

    @abstractmethod
    def __or__(self, other: t.Mapping[t.Any, t.Any]) -> "Metadata":
        """
        Returns a new metadata object with the contents of this metadata object updated with the contents of ``other``
        """

    @abstractmethod
    def setdefault(self, key: str, value: str) -> str:
        """
        If the header ``key`` does not exist, then set it to ``value``.
        Returns the header value.
        """

    @abstractmethod
    def update(self, other: t.Mapping[t.Any, t.Any]) -> None:
        """
        Sets all the headers in ``other`` in this object.

        For example, if this object is ``{"my-header": "my-value", "my-other-header": "my-other-value"}``
        and other is {"my-header": 3¸ "other-header": 4}
        """

    @abstractmethod
    def append(self, key: str, value: str) -> None:
        """
        Append a header, preserving any duplicate entries.
        """


class InferenceApiContext:
    request: "RequestContext"
    response: "ResponseContext"

    def __init__(self, request: "RequestContext", response: "ResponseContext"):
        self.request = request
        self.response = response

    @staticmethod
    def from_http(
        request: "starlette.requests.Request", response: "starlette.responses.Response"
    ) -> "InferenceApiContext":
        request_ctx = InferenceApiContext.RequestContext.from_http(request)
        response_ctx = InferenceApiContext.ResponseContext.from_http(response)

        return InferenceApiContext(request_ctx, response_ctx)

    class RequestContext:
        metadata: Metadata
        headers: Metadata

        def __init__(self, metadata: Metadata):
            self.metadata = metadata
            self.headers = metadata

        @staticmethod
        def from_http(
            request: "starlette.requests.Request",
        ) -> "InferenceApiContext.RequestContext":
            return InferenceApiContext.RequestContext(request.headers)  # type: ignore (coercing Starlette headers to Metadata)

    class ResponseContext:
        _raw_response: "starlette.responses.Response"
        metadata: Metadata
        headers: Metadata
        set_cookie: t.Callable[
            [
                str,
                str,
                int,
                int,
                str,
                str,
                bool,
                bool,
            ],
            None,
        ]
        delete_cookie: t.Callable[[str, str, str], None]

        def __init__(
            self,
            raw: "starlette.responses.Response",
            metadata: Metadata,
            set_cookie: t.Callable[
                [
                    str,
                    str,
                    int,
                    int,
                    str,
                    str,
                    bool,
                    bool,
                ],
                None,
            ],
            delete_cookie: t.Callable[[str, str, str], None],
        ):
            self._raw_response = raw
            self.metadata = metadata
            self.headers = metadata
            self.set_cookie = set_cookie
            self.delete_cookie = delete_cookie

        @property
        def status_code(self) -> int:
            return self._raw_response.status_code

        @status_code.setter
        def status_code(self, code: int) -> None:
            self._raw_response.status_code = code

        @staticmethod
        def from_http(
            response: "starlette.responses.Response",
        ) -> "InferenceApiContext.ResponseContext":
            return InferenceApiContext.ResponseContext(
                response,
                response.headers,  # type: ignore (coercing starlette Headers to Metadata)
                response.set_cookie,
                response.delete_cookie,
            )


class ServiceTraceContext:
    def __init__(self) -> None:
        self._request_id_var = contextvars.ContextVar(
            "_request_id_var", default=t.cast("t.Optional[int]", None)
        )

    @property
    def trace_id(self) -> t.Optional[int]:
        from opentelemetry import trace  # type: ignore

        span = trace.get_current_span()
        if span is None:
            return None
        return span.get_span_context().trace_id

    @property
    def span_id(self) -> t.Optional[int]:
        from opentelemetry import trace  # type: ignore

        span = trace.get_current_span()
        if span is None:
            return None
        return span.get_span_context().span_id

    @property
    def request_id(self) -> t.Optional[int]:
        """
        Different from span_id, request_id is unique for each inbound request.
        """
        return self._request_id_var.get()

    @request_id.setter
    def request_id(self, request_id: t.Optional[int]) -> None:
        self._request_id_var.set(request_id)

    @request_id.deleter
    def request_id(self) -> None:
        self._request_id_var.set(None)


trace_context = ServiceTraceContext()
