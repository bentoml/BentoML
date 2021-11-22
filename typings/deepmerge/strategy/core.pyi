"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable, List, Optional, Union

_StringOrFunction = Union[str, Callable[..., Any]]

STRATEGY_END: object = ...
class StrategyList:
    NAME: Optional[str] = ...
    def __init__(self, strategy_list: Union[_StringOrFunction, List[_StringOrFunction]]) -> None:
        ...

    @classmethod
    def _expand_strategy(cls, strategy: _StringOrFunction) -> _StringOrFunction:...
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...



