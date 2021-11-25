

from ..merger import Merger
from .core import StrategyList

class TypeConflictStrategies(StrategyList):
    """ contains the strategies provided for type conflicts. """
    @staticmethod
    def strategy_override(config: Merger, path: str, base: StrategyList, nxt: StrategyList) -> StrategyList:
        """ overrides the new object over the old object """
        ...

    @staticmethod
    def strategy_use_existing(config: Merger, path: str, base: StrategyList, nxt: StrategyList) -> StrategyList:
        """ uses the old object instead of the new object """
        ...

    @staticmethod
    def strategy_override_if_not_empty(config: Merger, path: str, base: StrategyList, nxt: StrategyList) -> StrategyList:
        """ overrides the new object over the old object only if the new object is not empty or null """
        ...



