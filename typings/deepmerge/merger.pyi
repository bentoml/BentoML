from typing import Any
from typing import TypeAlias

from .strategy.core import StrategyList
from .strategy.dict import DictStrategies
from .strategy.list import ListStrategies
from .strategy.set import SetStrategies

ConfigDictType: TypeAlias = dict[str, Any]

class Merger:
    PROVIDED_TYPE_STRATEGIES: dict[
        type, ListStrategies | DictStrategies | SetStrategies
    ] = ...

    def __init__(
        self,
        type_strategies: list[tuple[type, str]],
        fallback_strategies: list[str],
        type_conflict_strategies: list[str],
    ) -> None: ...
    def merge(self, base: ConfigDictType, nxt: ConfigDictType) -> None: ...
    def type_conflict_strategy(self, *args: Any) -> Any: ...
    def value_strategy(
        self, path: str, base: StrategyList, nxt: StrategyList
    ) -> None: ...
