from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable
from pandas._typing import AggFuncType, FrameOrSeries
from pandas.core.series import Series

if TYPE_CHECKING: ...

def reconstruct_func(
    func: AggFuncType | None, **kwargs
) -> tuple[bool, AggFuncType | None, list[str] | None, list[int] | None]: ...
def is_multi_agg_with_relabel(**kwargs) -> bool: ...
def normalize_keyword_aggregation(
    kwargs: dict,
) -> tuple[dict, list[str], list[int]]: ...
def maybe_mangle_lambdas(agg_spec: Any) -> Any: ...
def relabel_result(
    result: FrameOrSeries,
    func: dict[str, list[Callable | str]],
    columns: Iterable[Hashable],
    order: Iterable[int],
) -> dict[Hashable, Series]: ...
def validate_func_kwargs(
    kwargs: dict,
) -> tuple[list[str], list[str | Callable[..., Any]]]: ...
