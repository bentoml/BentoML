from __future__ import annotations

import typing as t
import inspect
import logging
import functools
from abc import ABC
from typing import TYPE_CHECKING
from collections.abc import Mapping

if TYPE_CHECKING:
    WrappedMethod = t.TypeVar("WrappedMethod", bound=t.Callable[..., t.Any])
    from ..types import AnyType

import attr

from ..types import LazyType
from ..utils import bentoml_cattr

logger = logging.getLogger(__name__)

RUNNABLE_METHOD_MARK: str = "_bentoml_runnable_method"
# not usable because attrs is unable to resolve the type at runtime
# BatchDimType: t.TypeAlias = tuple[list[int] | int, list[int] | int] | int
BatchDimType: t.TypeAlias = t.Union[t.Tuple[t.Union[t.List[int], int], int], int]


def batch_dim_structure_hook(
    batch_dim_encoded: int | list[list[int] | int], _
) -> BatchDimType:
    if isinstance(batch_dim_encoded, int):
        return batch_dim_encoded
    return tuple(batch_dim_encoded)


bentoml_cattr.register_structure_hook(BatchDimType, batch_dim_structure_hook)


class Runnable(ABC):
    SUPPORT_NVIDIA_GPU: bool
    SUPPORT_CPU_MULTI_THREADING: bool

    @classmethod
    def add_method(
        cls,
        method: t.Callable[..., t.Any],
        name: str,
        *,
        batchable: bool = False,
        batch_dim: BatchDimType = 0,
        input_spec: LazyType[t.Any] | t.Tuple[LazyType[t.Any], ...] | None = None,
        output_spec: LazyType[t.Any] | None = None,
    ):
        setattr(
            cls,
            name,
            Runnable.method(
                method,
                batchable=batchable,
                batch_dim=batch_dim,
                input_spec=input_spec,
                output_spec=output_spec,
            ),
        )

    @staticmethod
    def method(
        meth: WrappedMethod | None = None,
        *,
        batchable: bool = False,
        batch_dim: BatchDimType = 0,
        input_spec: AnyType | tuple[AnyType, ...] | None = None,
        output_spec: AnyType | None = None,
    ) -> t.Callable[[WrappedMethod], WrappedMethod] | WrappedMethod:
        def method_decorator(meth: WrappedMethod) -> WrappedMethod:
            params = inspect.signature(meth).parameters

            mapped_batch_dim = BatchDimMapping(batch_dim, params)

            setattr(
                meth,
                RUNNABLE_METHOD_MARK,
                RunnableMethodConfig(
                    batchable=batchable,
                    batch_dim=mapped_batch_dim,
                    input_spec=input_spec,
                    output_spec=output_spec,
                ),
            )
            return meth

        if callable(meth):
            return method_decorator(meth)
        return method_decorator

    @classmethod
    @functools.lru_cache(maxsize=1)
    def get_method_configs(cls) -> t.Dict[str, RunnableMethodConfig]:
        return {
            name: getattr(meth, RUNNABLE_METHOD_MARK)
            for name, meth in inspect.getmembers(
                cls, predicate=lambda x: hasattr(x, RUNNABLE_METHOD_MARK)
            )
        }


if TYPE_CHECKING:
    BatchDimSuper = Mapping[str | int, int]
else:
    BatchDimSuper = Mapping


class BatchDimMapping(BatchDimSuper):
    args: list[int]
    kwargs: dict[str, int]
    ret: int
    var_arg: int | None = None
    var_kw: int | None = None

    def __init__(
        self, batch_dim: BatchDimType, params: t.Mapping[str, inspect.Parameter]
    ):
        def get_batch_dim(idx: int) -> int:
            if isinstance(batch_dim, tuple):
                if idx == -1:
                    return batch_dim[1]
                elif isinstance(batch_dim[0], list):
                    return batch_dim[0][idx]
                else:
                    return batch_dim[0]
            else:
                return batch_dim

        self.args = []
        self.kwargs = {}
        for i, (name, param) in enumerate(params.items()):
            if param.kind in [
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ]:
                self.args.append(get_batch_dim(i))
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                self.kwargs[name] = get_batch_dim(i)
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                self.var_kw = get_batch_dim(i)
            else:
                self.var_arg = get_batch_dim(i)

        self.ret = get_batch_dim(-1)

    def __getitem__(self, key: str | int) -> int:
        if isinstance(key, int):
            if key < 0:
                return self.ret
            if key < len(self.args):
                return self.args[key]
            elif self.var_arg is not None:
                return self.var_arg
            else:
                raise KeyError(f"{key} not found")
        else:
            try:
                res = self.kwargs[key]
            except KeyError:
                if self.var_kw is not None:
                    res = self.var_kw
                else:
                    raise

            return res

    def __iter__(self) -> t.Iterator[str | int]:
        for i, _ in enumerate(self.args):
            yield i
        for key in self.kwargs:
            yield key

    def __len__(self) -> int:
        return len(self.args) + len(self.kwargs)


@attr.define()
class RunnableMethodConfig:
    batchable: bool
    batch_dim: BatchDimMapping
    input_spec: AnyType | t.Tuple[AnyType, ...] | None = None
    output_spec: AnyType | None = None
