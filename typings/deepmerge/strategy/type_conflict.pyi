"""
This type stub file was generated by pyright.
"""

from .core import StrategyList

class TypeConflictStrategies(StrategyList):
    """contains the strategies provided for type conflicts."""

    NAME = ...
    @staticmethod
    def strategy_override(config, path, base, nxt):
        """overrides the new object over the old object"""
        ...
    @staticmethod
    def strategy_use_existing(config, path, base, nxt):
        """uses the old object instead of the new object"""
        ...
    @staticmethod
    def strategy_override_if_not_empty(config, path, base, nxt):
        """overrides the new object over the old object only if the new object is not empty or null"""
        ...
