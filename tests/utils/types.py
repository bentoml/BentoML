import sys
import types
import typing as t

from bentoml._internal.models import Model
from bentoml._internal.types import Tag

if sys.version_info > (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


class Pipeline(Protocol):
    def __call__(
        self,
        model: t.Union[t.Callable[..., t.Any], t.Any],
        module: types.ModuleType,
        *args: t.Any,
        name: str = ...,
        metadata: t.Optional[t.Dict[str, t.Any]] = ...,
        return_model: bool = True,
        **kwargs: t.Any
    ) -> t.Union[Tag, Model]:
        ...


class InvalidModule(Protocol):
    def __call__(
        self, save_proc: t.Callable[..., t.Any], *args: t.Any, **kwargs: t.Any
    ) -> str:
        ...
