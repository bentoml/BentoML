
from decimal import Decimal
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Optional,
    Pattern,
    Tuple,
    Type,
    Union,
    overload,
)

from _pytest.compat import final

if TYPE_CHECKING:
    ...
class ApproxBase:
    """Provide shared utilities for making approximate comparisons between
    numbers or sequences of numbers."""
    __array_ufunc__ = ...
    __array_priority__ = ...
    def __init__(self, expected, rel=..., abs=..., nan_ok: bool = ...) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def __eq__(self, actual) -> bool:
        ...
    
    __hash__ = ...
    def __ne__(self, actual) -> bool:
        ...
    


class ApproxNumpy(ApproxBase):
    """Perform approximate comparisons where the expected value is numpy array."""
    def __repr__(self) -> str:
        ...
    
    def __eq__(self, actual) -> bool:
        ...
    


class ApproxMapping(ApproxBase):
    """Perform approximate comparisons where the expected value is a mapping
    with numeric values (the keys can be anything)."""
    def __repr__(self) -> str:
        ...
    
    def __eq__(self, actual) -> bool:
        ...
    


class ApproxSequencelike(ApproxBase):
    """Perform approximate comparisons where the expected value is a sequence of numbers."""
    def __repr__(self) -> str:
        ...
    
    def __eq__(self, actual) -> bool:
        ...
    


class ApproxScalar(ApproxBase):
    """Perform approximate comparisons where the expected value is a single number."""
    DEFAULT_ABSOLUTE_TOLERANCE: Union[float, Decimal] = ...
    DEFAULT_RELATIVE_TOLERANCE: Union[float, Decimal] = ...
    def __repr__(self) -> str:
        """Return a string communicating both the expected value and the
        tolerance for the comparison being made.

        For example, ``1.0 ± 1e-6``, ``(3+4j) ± 5e-6 ∠ ±180°``.
        """
        ...
    
    def __eq__(self, actual) -> bool:
        """Return whether the given value is equal to the expected value
        within the pre-specified tolerance."""
        ...
    
    __hash__ = ...
    @property
    def tolerance(self): # -> float | Decimal:
        """Return the tolerance for the comparison.

        This could be either an absolute tolerance or a relative tolerance,
        depending on what the user specified or which would be larger.
        """
        ...
    


class ApproxDecimal(ApproxScalar):
    """Perform approximate comparisons where the expected value is a Decimal."""
    DEFAULT_ABSOLUTE_TOLERANCE = ...
    DEFAULT_RELATIVE_TOLERANCE = ...


def approx(expected, rel=..., abs=..., nan_ok: bool = ...) -> ApproxBase:
    """Assert that two numbers (or two sets of numbers) are equal to each other
    within some tolerance.

    Due to the `intricacies of floating-point arithmetic`__, numbers that we
    would intuitively expect to be equal are not always so::

        >>> 0.1 + 0.2 == 0.3
        False

    __ https://docs.python.org/3/tutorial/floatingpoint.html

    This problem is commonly encountered when writing tests, e.g. when making
    sure that floating-point values are what you expect them to be.  One way to
    deal with this problem is to assert that two floating-point numbers are
    equal to within some appropriate tolerance::

        >>> abs((0.1 + 0.2) - 0.3) < 1e-6
        True

    However, comparisons like this are tedious to write and difficult to
    understand.  Furthermore, absolute comparisons like the one above are
    usually discouraged because there's no tolerance that works well for all
    situations.  ``1e-6`` is good for numbers around ``1``, but too small for
    very big numbers and too big for very small ones.  It's better to express
    the tolerance as a fraction of the expected value, but relative comparisons
    like that are even more difficult to write correctly and concisely.

    The ``approx`` class performs floating-point comparisons using a syntax
    that's as intuitive as possible::

        >>> from pytest import approx
        >>> 0.1 + 0.2 == approx(0.3)
        True

    The same syntax also works for sequences of numbers::

        >>> (0.1 + 0.2, 0.2 + 0.4) == approx((0.3, 0.6))
        True

    Dictionary *values*::

        >>> {'a': 0.1 + 0.2, 'b': 0.2 + 0.4} == approx({'a': 0.3, 'b': 0.6})
        True

    ``numpy`` arrays::

        >>> import numpy as np                                                          # doctest: +SKIP
        >>> np.array([0.1, 0.2]) + np.array([0.2, 0.4]) == approx(np.array([0.3, 0.6])) # doctest: +SKIP
        True

    And for a ``numpy`` array against a scalar::

        >>> import numpy as np                                         # doctest: +SKIP
        >>> np.array([0.1, 0.2]) + np.array([0.2, 0.1]) == approx(0.3) # doctest: +SKIP
        True

    By default, ``approx`` considers numbers within a relative tolerance of
    ``1e-6`` (i.e. one part in a million) of its expected value to be equal.
    This treatment would lead to surprising results if the expected value was
    ``0.0``, because nothing but ``0.0`` itself is relatively close to ``0.0``.
    To handle this case less surprisingly, ``approx`` also considers numbers
    within an absolute tolerance of ``1e-12`` of its expected value to be
    equal.  Infinity and NaN are special cases.  Infinity is only considered
    equal to itself, regardless of the relative tolerance.  NaN is not
    considered equal to anything by default, but you can make it be equal to
    itself by setting the ``nan_ok`` argument to True.  (This is meant to
    facilitate comparing arrays that use NaN to mean "no data".)

    Both the relative and absolute tolerances can be changed by passing
    arguments to the ``approx`` constructor::

        >>> 1.0001 == approx(1)
        False
        >>> 1.0001 == approx(1, rel=1e-3)
        True
        >>> 1.0001 == approx(1, abs=1e-3)
        True

    If you specify ``abs`` but not ``rel``, the comparison will not consider
    the relative tolerance at all.  In other words, two numbers that are within
    the default relative tolerance of ``1e-6`` will still be considered unequal
    if they exceed the specified absolute tolerance.  If you specify both
    ``abs`` and ``rel``, the numbers will be considered equal if either
    tolerance is met::

        >>> 1 + 1e-8 == approx(1)
        True
        >>> 1 + 1e-8 == approx(1, abs=1e-12)
        False
        >>> 1 + 1e-8 == approx(1, rel=1e-6, abs=1e-12)
        True

    You can also use ``approx`` to compare nonnumeric types, or dicts and
    sequences containing nonnumeric types, in which case it falls back to
    strict equality. This can be useful for comparing dicts and sequences that
    can contain optional values::

        >>> {"required": 1.0000005, "optional": None} == approx({"required": 1, "optional": None})
        True
        >>> [None, 1.0000005] == approx([None,1])
        True
        >>> ["foo", 1.0000005] == approx([None,1])
        False

    If you're thinking about using ``approx``, then you might want to know how
    it compares to other good ways of comparing floating-point numbers.  All of
    these algorithms are based on relative and absolute tolerances and should
    agree for the most part, but they do have meaningful differences:

    - ``math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)``:  True if the relative
      tolerance is met w.r.t. either ``a`` or ``b`` or if the absolute
      tolerance is met.  Because the relative tolerance is calculated w.r.t.
      both ``a`` and ``b``, this test is symmetric (i.e.  neither ``a`` nor
      ``b`` is a "reference value").  You have to specify an absolute tolerance
      if you want to compare to ``0.0`` because there is no tolerance by
      default.  `More information...`__

      __ https://docs.python.org/3/library/math.html#math.isclose

    - ``numpy.isclose(a, b, rtol=1e-5, atol=1e-8)``: True if the difference
      between ``a`` and ``b`` is less that the sum of the relative tolerance
      w.r.t. ``b`` and the absolute tolerance.  Because the relative tolerance
      is only calculated w.r.t. ``b``, this test is asymmetric and you can
      think of ``b`` as the reference value.  Support for comparing sequences
      is provided by ``numpy.allclose``.  `More information...`__

      __ https://numpy.org/doc/stable/reference/generated/numpy.isclose.html

    - ``unittest.TestCase.assertAlmostEqual(a, b)``: True if ``a`` and ``b``
      are within an absolute tolerance of ``1e-7``.  No relative tolerance is
      considered and the absolute tolerance cannot be changed, so this function
      is not appropriate for very large or very small numbers.  Also, it's only
      available in subclasses of ``unittest.TestCase`` and it's ugly because it
      doesn't follow PEP8.  `More information...`__

      __ https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertAlmostEqual

    - ``a == pytest.approx(b, rel=1e-6, abs=1e-12)``: True if the relative
      tolerance is met w.r.t. ``b`` or if the absolute tolerance is met.
      Because the relative tolerance is only calculated w.r.t. ``b``, this test
      is asymmetric and you can think of ``b`` as the reference value.  In the
      special case that you explicitly specify an absolute tolerance but not a
      relative tolerance, only the absolute tolerance is considered.

    .. warning::

       .. versionchanged:: 3.2

       In order to avoid inconsistent behavior, ``TypeError`` is
       raised for ``>``, ``>=``, ``<`` and ``<=`` comparisons.
       The example below illustrates the problem::

           assert approx(0.1) > 0.1 + 1e-10  # calls approx(0.1).__gt__(0.1 + 1e-10)
           assert 0.1 + 1e-10 > approx(0.1)  # calls approx(0.1).__lt__(0.1 + 1e-10)

       In the second example one expects ``approx(0.1).__le__(0.1 + 1e-10)``
       to be called. But instead, ``approx(0.1).__lt__(0.1 + 1e-10)`` is used to
       comparison. This is because the call hierarchy of rich comparisons
       follows a fixed behavior. `More information...`__

       __ https://docs.python.org/3/reference/datamodel.html#object.__ge__

    .. versionchanged:: 3.7.1
       ``approx`` raises ``TypeError`` when it encounters a dict value or
       sequence element of nonnumeric type.

    .. versionchanged:: 6.1.0
       ``approx`` falls back to strict equality for nonnumeric types instead
       of raising ``TypeError``.
    """
    ...

_E = ...
@overload
def raises(expected_exception: Union[Type[_E], Tuple[Type[_E], ...]], *, match: Optional[Union[str, Pattern[str]]] = ...) -> RaisesContext[_E]:
    ...

@overload
def raises(expected_exception: Union[Type[_E], Tuple[Type[_E], ...]], func: Callable[..., Any], *args: Any, **kwargs: Any) -> _pytest._code.ExceptionInfo[_E]:
    ...

def raises(expected_exception: Union[Type[_E], Tuple[Type[_E], ...]], *args: Any, **kwargs: Any) -> Union[RaisesContext[_E], _pytest._code.ExceptionInfo[_E]]:
    r"""Assert that a code block/function call raises ``expected_exception``
    or raise a failure exception otherwise.

    :kwparam match:
        If specified, a string containing a regular expression,
        or a regular expression object, that is tested against the string
        representation of the exception using ``re.search``. To match a literal
        string that may contain `special characters`__, the pattern can
        first be escaped with ``re.escape``.

        (This is only used when ``pytest.raises`` is used as a context manager,
        and passed through to the function otherwise.
        When using ``pytest.raises`` as a function, you can use:
        ``pytest.raises(Exc, func, match="passed on").match("my pattern")``.)

        __ https://docs.python.org/3/library/re.html#regular-expression-syntax

    .. currentmodule:: _pytest._code

    Use ``pytest.raises`` as a context manager, which will capture the exception of the given
    type::

        >>> import pytest
        >>> with pytest.raises(ZeroDivisionError):
        ...    1/0

    If the code block does not raise the expected exception (``ZeroDivisionError`` in the example
    above), or no exception at all, the check will fail instead.

    You can also use the keyword argument ``match`` to assert that the
    exception matches a text or regex::

        >>> with pytest.raises(ValueError, match='must be 0 or None'):
        ...     raise ValueError("value must be 0 or None")

        >>> with pytest.raises(ValueError, match=r'must be \d+$'):
        ...     raise ValueError("value must be 42")

    The context manager produces an :class:`ExceptionInfo` object which can be used to inspect the
    details of the captured exception::

        >>> with pytest.raises(ValueError) as exc_info:
        ...     raise ValueError("value must be 42")
        >>> assert exc_info.type is ValueError
        >>> assert exc_info.value.args[0] == "value must be 42"

    .. note::

       When using ``pytest.raises`` as a context manager, it's worthwhile to
       note that normal context manager rules apply and that the exception
       raised *must* be the final line in the scope of the context manager.
       Lines of code after that, within the scope of the context manager will
       not be executed. For example::

           >>> value = 15
           >>> with pytest.raises(ValueError) as exc_info:
           ...     if value > 10:
           ...         raise ValueError("value must be <= 10")
           ...     assert exc_info.type is ValueError  # this will not execute

       Instead, the following approach must be taken (note the difference in
       scope)::

           >>> with pytest.raises(ValueError) as exc_info:
           ...     if value > 10:
           ...         raise ValueError("value must be <= 10")
           ...
           >>> assert exc_info.type is ValueError

    **Using with** ``pytest.mark.parametrize``

    When using :ref:`pytest.mark.parametrize ref`
    it is possible to parametrize tests such that
    some runs raise an exception and others do not.

    See :ref:`parametrizing_conditional_raising` for an example.

    **Legacy form**

    It is possible to specify a callable by passing a to-be-called lambda::

        >>> raises(ZeroDivisionError, lambda: 1/0)
        <ExceptionInfo ...>

    or you can specify an arbitrary callable with arguments::

        >>> def f(x): return 1/x
        ...
        >>> raises(ZeroDivisionError, f, 0)
        <ExceptionInfo ...>
        >>> raises(ZeroDivisionError, f, x=0)
        <ExceptionInfo ...>

    The form above is fully supported but discouraged for new code because the
    context manager form is regarded as more readable and less error-prone.

    .. note::
        Similar to caught exception objects in Python, explicitly clearing
        local references to returned ``ExceptionInfo`` objects can
        help the Python interpreter speed up its garbage collection.

        Clearing those references breaks a reference cycle
        (``ExceptionInfo`` --> caught exception --> frame stack raising
        the exception --> current frame stack --> local variables -->
        ``ExceptionInfo``) which makes Python keep all objects referenced
        from that cycle (including all local variables in the current
        frame) alive until the next cyclic garbage collection run.
        More detailed information can be found in the official Python
        documentation for :ref:`the try statement <python:try>`.
    """
    ...

@final
class RaisesContext(Generic[_E]):
    def __init__(self, expected_exception: Union[Type[_E], Tuple[Type[_E], ...]], message: str, match_expr: Optional[Union[str, Pattern[str]]] = ...) -> None:
        ...
    
    def __enter__(self) -> _pytest._code.ExceptionInfo[_E]:
        ...
    
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> bool:
        ...
    


