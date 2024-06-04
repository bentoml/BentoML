from __future__ import annotations

import contextlib
import contextvars
import os
import shutil
import tempfile
import typing as t
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import attr
import starlette.datastructures

from .utils.http import Cookie

if TYPE_CHECKING:
    import starlette.requests
    import starlette.responses

_request_var: contextvars.ContextVar[starlette.requests.Request] = (
    contextvars.ContextVar("request")
)
_response_var: contextvars.ContextVar[ServiceContext.ResponseContext] = (
    contextvars.ContextVar("response")
)


def request_temp_dir() -> str:
    """A request-unique directory for storing temporary files"""
    request = _request_var.get()
    if "temp_dir" not in request.state:
        request.state["temp_dir"] = tempfile.mkdtemp(prefix="bentoml-request-")
    return t.cast(str, request.state["temp_dir"])


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


class ServiceContext:
    def __init__(self) -> None:
        # A dictionary for storing global state shared by the process
        self.state: dict[str, t.Any] = {}

    @contextlib.contextmanager
    def in_request(
        self, request: starlette.requests.Request
    ) -> t.Generator[ServiceContext, None, None]:
        request.metadata = request.headers  # type: ignore[attr-defined]
        request_token = _request_var.set(request)
        response_token = _response_var.set(ServiceContext.ResponseContext())
        try:
            yield self
        finally:
            if "temp_dir" in request.state:
                shutil.rmtree(
                    t.cast(str, request.state["temp_dir"]), ignore_errors=True
                )
            _request_var.reset(request_token)
            _response_var.reset(response_token)

    @property
    def request(self) -> starlette.requests.Request:
        return _request_var.get()

    @property
    def response(self) -> ResponseContext:
        return _response_var.get()

    @property
    def temp_dir(self) -> str:
        return request_temp_dir()

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
    _service_name_var = contextvars.ContextVar(
        "_service_name_var", default=t.cast("t.Optional[str]", None)
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
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None:
            return 0
        return 1 if span.get_span_context().trace_flags.sampled else 0

    @property
    def span_id(self) -> t.Optional[int]:
        from opentelemetry import trace

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

    @property
    def service_name(self) -> t.Optional[str]:
        return self._service_name_var.get()

    @service_name.setter
    def service_name(self, service_name: t.Optional[str]) -> None:
        self._service_name_var.set(service_name)


@attr.define
class _ComponentContext:
    bento_name: str = ""
    bento_version: str = "not available"
    service_type: str | None = None
    service_name: str | None = None
    worker_index: int | None = None

    @property
    def component_type(self) -> str | None:
        return self.service_type

    @property
    def component_name(self) -> str | None:
        return self.service_name

    @property
    def component_index(self) -> int | None:
        return self.worker_index

    @property
    def yatai_bento_deployment_name(self) -> str:
        return os.environ.get("YATAI_BENTO_DEPLOYMENT_NAME", "")

    @property
    def yatai_bento_deployment_namespace(self) -> str:
        return os.environ.get("YATAI_BENTO_DEPLOYMENT_NAMESPACE", "")


trace_context = _ServiceTraceContext()
server_context = component_context = _ComponentContext()
