from .core import StrategyList
from ..merger import Merger

class ListStrategies(StrategyList):
    NAME: str = ...

    @staticmethod
    def strategy_override(
        config: Merger, path: str, base: StrategyList, nxt: StrategyList
    ) -> StrategyList: ...
    @staticmethod
    def strategy_prepend(
        config: Merger, path: str, base: StrategyList, nxt: StrategyList
    ) -> StrategyList: ...
    @staticmethod
    def strategy_append(
        config: Merger, path: str, base: StrategyList, nxt: StrategyList
    ) -> StrategyList: ...
