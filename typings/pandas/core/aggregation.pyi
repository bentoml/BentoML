from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable

from pandas._typing import AggFuncType, FrameOrSeries
from pandas.core.series import Series

"""
aggregation.py contains utility functions to handle multiple named and lambda
kwarg aggregations in groupby and DataFrame/Series aggregation
"""
if TYPE_CHECKING: ...

def reconstruct_func(
    func: AggFuncType | None, **kwargs
) -> tuple[bool, AggFuncType | None, list[str] | None, list[int] | None]:
    """
    This is the internal function to reconstruct func given if there is relabeling
    or not and also normalize the keyword to get new order of columns.

    If named aggregation is applied, `func` will be None, and kwargs contains the
    column and aggregation function information to be parsed;
    If named aggregation is not applied, `func` is either string (e.g. 'min') or
    Callable, or list of them (e.g. ['min', np.max]), or the dictionary of column name
    and str/Callable/list of them (e.g. {'A': 'min'}, or {'A': [np.min, lambda x: x]})

    If relabeling is True, will return relabeling, reconstructed func, column
    names, and the reconstructed order of columns.
    If relabeling is False, the columns and order will be None.

    Parameters
    ----------
    func: agg function (e.g. 'min' or Callable) or list of agg functions
        (e.g. ['min', np.max]) or dictionary (e.g. {'A': ['min', np.max]}).
    **kwargs: dict, kwargs used in is_multi_agg_with_relabel and
        normalize_keyword_aggregation function for relabelling

    Returns
    -------
    relabelling: bool, if there is relabelling or not
    func: normalized and mangled func
    columns: list of column names
    order: list of columns indices

    Examples
    --------
    >>> reconstruct_func(None, **{"foo": ("col", "min")})
    (True, defaultdict(<class 'list'>, {'col': ['min']}), ('foo',), array([0]))

    >>> reconstruct_func("min")
    (False, 'min', None, None)
    """
    ...

def is_multi_agg_with_relabel(**kwargs) -> bool:
    """
    Check whether kwargs passed to .agg look like multi-agg with relabeling.

    Parameters
    ----------
    **kwargs : dict

    Returns
    -------
    bool

    Examples
    --------
    >>> is_multi_agg_with_relabel(a="max")
    False
    >>> is_multi_agg_with_relabel(a_max=("a", "max"), a_min=("a", "min"))
    True
    >>> is_multi_agg_with_relabel()
    False
    """
    ...

def normalize_keyword_aggregation(kwargs: dict) -> tuple[dict, list[str], list[int]]:
    """
    Normalize user-provided "named aggregation" kwargs.
    Transforms from the new ``Mapping[str, NamedAgg]`` style kwargs
    to the old Dict[str, List[scalar]]].

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    aggspec : dict
        The transformed kwargs.
    columns : List[str]
        The user-provided keys.
    col_idx_order : List[int]
        List of columns indices.

    Examples
    --------
    >>> normalize_keyword_aggregation({"output": ("input", "sum")})
    (defaultdict(<class 'list'>, {'input': ['sum']}), ('output',), array([0]))
    """
    ...

def maybe_mangle_lambdas(agg_spec: Any) -> Any:
    """
    Make new lambdas with unique names.

    Parameters
    ----------
    agg_spec : Any
        An argument to GroupBy.agg.
        Non-dict-like `agg_spec` are pass through as is.
        For dict-like `agg_spec` a new spec is returned
        with name-mangled lambdas.

    Returns
    -------
    mangled : Any
        Same type as the input.

    Examples
    --------
    >>> maybe_mangle_lambdas('sum')
    'sum'
    >>> maybe_mangle_lambdas([lambda: 1, lambda: 2])  # doctest: +SKIP
    [<function __main__.<lambda_0>,
     <function pandas...._make_lambda.<locals>.f(*args, **kwargs)>]
    """
    ...

def relabel_result(
    result: FrameOrSeries,
    func: dict[str, list[Callable | str]],
    columns: Iterable[Hashable],
    order: Iterable[int],
) -> dict[Hashable, Series]:
    """
    Internal function to reorder result if relabelling is True for
    dataframe.agg, and return the reordered result in dict.

    Parameters:
    ----------
    result: Result from aggregation
    func: Dict of (column name, funcs)
    columns: New columns name for relabelling
    order: New order for relabelling

    Examples:
    ---------
    >>> result = DataFrame({"A": [np.nan, 2, np.nan],
    ...       "C": [6, np.nan, np.nan], "B": [np.nan, 4, 2.5]})  # doctest: +SKIP
    >>> funcs = {"A": ["max"], "C": ["max"], "B": ["mean", "min"]}
    >>> columns = ("foo", "aab", "bar", "dat")
    >>> order = [0, 1, 2, 3]
    >>> _relabel_result(result, func, columns, order)  # doctest: +SKIP
    dict(A=Series([2.0, NaN, NaN, NaN], index=["foo", "aab", "bar", "dat"]),
         C=Series([NaN, 6.0, NaN, NaN], index=["foo", "aab", "bar", "dat"]),
         B=Series([NaN, NaN, 2.5, 4.0], index=["foo", "aab", "bar", "dat"]))
    """
    ...

def validate_func_kwargs(
    kwargs: dict,
) -> tuple[list[str], list[str | Callable[..., Any]]]:
    """
    Validates types of user-provided "named aggregation" kwargs.
    `TypeError` is raised if aggfunc is not `str` or callable.

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    columns : List[str]
        List of user-provied keys.
    func : List[Union[str, callable[...,Any]]]
        List of user-provided aggfuncs

    Examples
    --------
    >>> validate_func_kwargs({'one': 'min', 'two': 'max'})
    (['one', 'two'], ['min', 'max'])
    """
    ...
