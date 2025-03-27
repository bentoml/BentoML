from __future__ import annotations

import abc
import functools
import typing as t
from http import HTTPStatus

import attrs
import httpx

from _bentoml_sdk import IODescriptor
from bentoml.exceptions import BentoMLException

T = t.TypeVar("T")


def map_exception(resp: httpx.Response) -> BentoMLException:
    status = HTTPStatus(resp.status_code)
    exc = BentoMLException.error_mapping.get(status, BentoMLException)
    return exc(resp.text, error_code=status)


@attrs.define(slots=True)
class ClientEndpoint:
    name: str
    route: str
    doc: str | None = None
    input: dict[str, t.Any] = attrs.field(factory=dict)
    output: dict[str, t.Any] = attrs.field(factory=dict)
    input_spec: type[IODescriptor] | None = None
    output_spec: type[IODescriptor] | None = None
    stream_output: bool = False
    is_task: bool = False


class AbstractClient(abc.ABC):
    endpoints: dict[str, ClientEndpoint]

    def _setup_endpoints(self) -> None:
        self._setup_done = True
        for name in self.endpoints:
            if name == "__call__":
                # __call__ must be set on the class
                continue
            attr_name = name
            if getattr(self, attr_name, None) is not None:
                attr_name = f"api_{name}"  # prefix to avoid name conflict
            setattr(self, attr_name, self._make_method(name))

    def _make_method(self, name: str) -> t.Callable[..., t.Any]:
        endpoint = self.endpoints[name]

        def method(*args: t.Any, **kwargs: t.Any) -> t.Any:
            return self.call(name, *args, **kwargs)

        method.__doc__ = endpoint.doc
        if endpoint.input_spec is not None:
            method.__annotations__ = endpoint.input_spec.__annotations__
            method.__signature__ = endpoint.input_spec.__signature__
        if endpoint.is_task:
            method.submit = functools.partial(self._submit, endpoint)
            method.get = functools.partial(self._get_task_result, endpoint)
            method.get_status = functools.partial(self._get_task_status, endpoint)
            method.cancel = functools.partial(self._cancel_task, endpoint)
            method.retry = functools.partial(self._retry_task, endpoint)
        return method

    @abc.abstractmethod
    def call(self, __name: str, /, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Call a service method by its name.
        It takes the same arguments as the service method.
        """

    @abc.abstractmethod
    def _submit(
        self, __endpoint: ClientEndpoint, /, *args: t.Any, **kwargs: t.Any
    ) -> t.Any:
        """Submit a job to the service.
        It takes the same arguments as the service method.
        """

    def _get_task_status(self, __endpoint: ClientEndpoint, /, task_id: str) -> t.Any:
        """Get the status of a task."""

    def _cancel_task(self, __endpoint: ClientEndpoint, /, task_id: str) -> t.Any:
        """Cancel a task."""

    def _get_task_result(self, __endpoint: ClientEndpoint, /, task_id: str) -> t.Any:
        """Get the result of a task."""

    def _retry_task(self, __endpoint: ClientEndpoint, /, task_id: str) -> t.Any:
        """Retry a task."""

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if "__call__" not in self.endpoints:
            raise TypeError("This service is not callable.")
        return self.call("__call__", *args, **kwargs)
