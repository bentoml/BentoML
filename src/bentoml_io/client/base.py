from __future__ import annotations

import abc
import typing as t

T = t.TypeVar("T")


class AbstractClient(abc.ABC):
    @abc.abstractmethod
    def call(self, __name: str, /, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Call a service method by its name.
        It takes the same arguments as the service method.
        """
