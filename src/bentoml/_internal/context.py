from __future__ import annotations

import os
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
        query_params: Metadata

        def __init__(self, metadata: Metadata, query_params: Metadata):
            self.metadata = metadata
            self.headers = metadata
            self.query_params = query_params

        @staticmethod
        def from_http(
            request: "starlette.requests.Request",
        ) -> "InferenceApiContext.RequestContext":
            return InferenceApiContext.RequestContext(
                request.headers,  # type: ignore # coercing Starlette types to Metadata
                request.query_params,  # type: ignore
            )

    @attr.define
    class ResponseContext:
        metadata: Metadata
        cookies: list[Cookie]
        headers: Metadata
        status_code: int

        def __init__(self):
            self.metadata = starlette.datastructures.MutableHeaders()  # type: ignore (coercing Starlette headers to Metadata)
            self.headers = self.metadata  # type: ignore (coercing Starlette headers to Metadata)
            self.cookies = []
            self.status_code = 200

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


class _ServiceTraceContext:
    _request_id_var = contextvars.ContextVar(
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
    def sampled(self) -> int:
        from opentelemetry import trace  # type: ignore

        span = trace.get_current_span()
        if span is None:
            return 0
        return 1 if span.get_span_context().trace_flags.sampled else 0

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


class _ComponentContext:
    bento_name: str = ""
    bento_version: str = "not available"
    component_type: str | None = None
    component_name: str | None = None
    component_index: int | None = None

    @property
    def yatai_bento_deployment_name(self) -> str:
        return os.environ.get("YATAI_BENTO_DEPLOYMENT_NAME", "")

    @property
    def yatai_bento_deployment_namespace(self) -> str:
        return os.environ.get("YATAI_BENTO_DEPLOYMENT_NAMESPACE", "")


trace_context = _ServiceTraceContext()
component_context = _ComponentContext()
