from __future__ import annotations

import abc
import typing as t

T = t.TypeVar("T")


class AbstractClient(abc.ABC):
    @abc.abstractmethod
    def call(self, name: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Call a service method by its name.
        It takes the same arguments as the service method.
        """

    async def __aenter__(self: T) -> T:
        return self

    async def __aexit__(self, *args: t.Any) -> None:
        pass
