import abc

"""
Engine classes for :func:`~pandas.eval`
"""
_ne_builtins = ...

class NumExprClobberingError(NameError): ...

class AbstractEngine(metaclass=abc.ABCMeta):
    """Object serving as a base class for all engines."""

    has_neg_frac = ...
    def __init__(self, expr) -> None: ...
    def convert(self) -> str:
        """
        Convert an expression for evaluation.

        Defaults to return the expression as a string.
        """
        ...
    def evaluate(self) -> object:
        """
        Run the engine on the expression.

        This method performs alignment which is necessary no matter what engine
        is being used, thus its implementation is in the base class.

        Returns
        -------
        object
            The result of the passed expression.
        """
        ...

class NumExprEngine(AbstractEngine):
    """NumExpr engine class"""

    has_neg_frac = ...

class PythonEngine(AbstractEngine):
    """
    Evaluate an expression in Python space.

    Mostly for testing purposes.
    """

    has_neg_frac = ...
    def evaluate(self): ...

ENGINES: dict[str, type[AbstractEngine]] = ...
