from ..merger import Merger
from .core import StrategyList

class TypeConflictStrategies(StrategyList):
    @staticmethod
    def strategy_override(
        config: Merger, path: str, base: StrategyList, nxt: StrategyList
    ) -> StrategyList: ...
    @staticmethod
    def strategy_use_existing(
        config: Merger, path: str, base: StrategyList, nxt: StrategyList
    ) -> StrategyList: ...
    @staticmethod
    def strategy_override_if_not_empty(
        config: Merger, path: str, base: StrategyList, nxt: StrategyList
    ) -> StrategyList: ...
