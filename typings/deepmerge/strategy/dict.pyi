

from ..merger import Merger
from .core import StrategyList

class DictStrategies(StrategyList):
    """
    Contains the strategies provided for dictionaries.

    """
    @staticmethod
    def strategy_merge(config: Merger, path: str, base: StrategyList, nxt: StrategyList)-> StrategyList:
        """
        for keys that do not exists,
        use them directly. if the key exists
        in both dictionaries, attempt a value merge.
        """
        ...
    
    @staticmethod
    def strategy_override(config: Merger, path: str, base: StrategyList, nxt: StrategyList)->StrategyList:
        """
        move all keys in nxt into base, overriding
        conflicts.
        """
        ...
    


