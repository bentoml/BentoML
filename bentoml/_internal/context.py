from __future__ import annotations

import typing as t
import contextvars
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import attr
import starlette.datastructures

from .utils.http import Cookie

if TYPE_CHECKING:
    import starlette.requests
    import starlette.responses


class Metadata(t.Mapping[str, str], ABC):
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
    def __ior__(self, other: t.Mapping[t.Any, t.Any]) -> Metadata:
        """
        Updates this metadata with the contents of ``other``
        """

    @abstractmethod
    def __or__(self, other: t.Mapping[t.Any, t.Any]) -> Metadata:
        """
        Returns a new metadata object with the contents of this metadata object updated with the contents of ``other``
        """

    @abstractmethod
    def update(self, other: t.Mapping[t.Any, t.Any]) -> None:
        """
        Sets all the headers in ``other`` in this object.

        For example, if this object is ``{"my-header": "my-value", "my-other-header": "my-other-value"}``
        and other is {"my-header": 3¸ "other-header": 4}
        """

    @abstractmethod
    def setdefault(self, key: str, value: str) -> str:
        """
        If the header ``key`` does not exist, then set it to ``value``.
        Returns the header value.
        """

    @abstractmethod
    def append(self, key: str, value: str) -> None:
        """
        Append a header, preserving any duplicate entries.
        """

    @abstractmethod
    def mutablecopy(self) -> Metadata:
        """
        Returns a copy of this metadata object.
        """


@attr.define
class InferenceApiContext:
    request: "RequestContext"
    response: "ResponseContext"

    def __init__(self, request: "RequestContext", response: "ResponseContext"):
        self.request = request
        self.response = response

    @staticmethod
    def from_http(request: "starlette.requests.Request") -> "InferenceApiContext":
        request_ctx = InferenceApiContext.RequestContext.from_http(request)
        response_ctx = InferenceApiContext.ResponseContext()

        return InferenceApiContext(request_ctx, response_ctx)

    @attr.define
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

    @attr.define
    class ResponseContext:
        metadata: Metadata
        cookies: list[Cookie]
        headers: Metadata
        status_code: int = 200

        def __init__(self):
            self.metadata = starlette.datastructures.MutableHeaders()  # type: ignore (coercing Starlette headers to Metadata)
            self.headers = self.metadata  # type: ignore (coercing Starlette headers to Metadata)

        def set_cookie(
            self,
            key: str,
            value: str,
            max_age: int | None = None,
            expires: int | None = None,
            path: str = "/",
            domain: str | None = None,
            secure: bool = False,
            httponly: bool = False,
            samesite: str = "lax",
        ):
            self.cookies.append(
                Cookie(
                    key,
                    value,
                    max_age,
                    expires,
                    path,
                    domain,
                    secure,
                    httponly,
                    samesite,
                )
            )


class ServiceTraceContext:
    def __init__(self) -> None:
        self._request_id_var = contextvars.ContextVar(
            "_request_id_var", default=t.cast("t.Optional[int]", None)
        )

    @property
    def trace_id(self) -> t.Optional[int]:
        from opentelemetry import trace

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
