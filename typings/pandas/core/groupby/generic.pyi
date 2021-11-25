from typing import Any, Callable, Union

from pandas._typing import FrameOrSeries
from pandas.core.frame import DataFrame
from pandas.core.groupby import base
from pandas.core.groupby.groupby import (
    GroupBy,
    _agg_template,
    _apply_docs,
    _transform_template,
)
from pandas.core.series import Series
from pandas.util._decorators import Appender, Substitution, doc

"""
Define the SeriesGroupBy and DataFrameGroupBy
classes that hold the groupby interfaces (and some implementations).

These are user facing as the result of the ``df.groupby(...)`` operations,
which here returns a DataFrameGroupBy object.
"""
NamedAgg = ...
AggScalar = Union[str, Callable[..., Any]]
ScalarResult = ...

def generate_property(name: str, klass: type[FrameOrSeries]):  # -> property:
    """
    Create a property for a GroupBy subclass to dispatch to DataFrame/Series.

    Parameters
    ----------
    name : str
    klass : {DataFrame, Series}

    Returns
    -------
    property
    """
    ...

def pin_allowlisted_properties(
    klass: type[FrameOrSeries], allowlist: frozenset[str]
):  # -> (cls: Unknown) -> Unknown:
    """
    Create GroupBy member defs for DataFrame/Series names in a allowlist.

    Parameters
    ----------
    klass : DataFrame or Series class
        class where members are defined.
    allowlist : frozenset[str]
        Set of names of klass methods to be constructed

    Returns
    -------
    class decorator

    Notes
    -----
    Since we don't want to override methods explicitly defined in the
    base class, any such name is skipped.
    """
    ...

@pin_allowlisted_properties(Series, base.series_apply_allowlist)
class SeriesGroupBy(GroupBy[Series]):
    _apply_allowlist = ...
    _agg_examples_doc = ...
    @Appender(
        _apply_docs["template"].format(
            input="series", examples=_apply_docs["series_examples"]
        )
    )
    def apply(self, func, *args, **kwargs): ...
    @doc(_agg_template, examples=_agg_examples_doc, klass="Series")
    def aggregate(self, func=..., *args, engine=..., engine_kwargs=..., **kwargs): ...
    agg = ...
    @Substitution(klass="Series")
    @Appender(_transform_template)
    def transform(self, func, *args, engine=..., engine_kwargs=..., **kwargs): ...
    def filter(self, func, dropna: bool = ..., *args, **kwargs):  # -> Any:
        """
        Return a copy of a Series excluding elements from groups that
        do not satisfy the boolean criterion specified by func.

        Parameters
        ----------
        func : function
            To apply to each group. Should return True or False.
        dropna : Drop groups that do not pass the filter. True by default;
            if False, groups that evaluate False are filled with NaNs.

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
        ...                           'foo', 'bar'],
        ...                    'B' : [1, 2, 3, 4, 5, 6],
        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]})
        >>> grouped = df.groupby('A')
        >>> df.groupby('A').B.filter(lambda x: x.mean() > 3.)
        1    2
        3    4
        5    6
        Name: B, dtype: int64

        Returns
        -------
        filtered : Series
        """
        ...
    def nunique(self, dropna: bool = ...) -> Series:
        """
        Return number of unique elements in the group.

        Returns
        -------
        Series
            Number of unique values within each group.
        """
        ...
    @doc(Series.describe)
    def describe(self, **kwargs): ...
    def value_counts(
        self,
        normalize: bool = ...,
        sort: bool = ...,
        ascending: bool = ...,
        bins=...,
        dropna: bool = ...,
    ): ...
    def count(self) -> Series:
        """
        Compute count of group, excluding missing values.

        Returns
        -------
        Series
            Count of values within each group.
        """
        ...
    def pct_change(self, periods=..., fill_method=..., limit=..., freq=...):  # -> Any:
        """Calculate pct_change of each value to previous entry in group"""
        ...

@pin_allowlisted_properties(DataFrame, base.dataframe_apply_allowlist)
class DataFrameGroupBy(GroupBy[DataFrame]):
    _apply_allowlist = ...
    _agg_examples_doc = ...
    @doc(_agg_template, examples=_agg_examples_doc, klass="DataFrame")
    def aggregate(self, func=..., *args, engine=..., engine_kwargs=..., **kwargs): ...
    agg = ...
    @Substitution(klass="DataFrame")
    @Appender(_transform_template)
    def transform(self, func, *args, engine=..., engine_kwargs=..., **kwargs): ...
    def filter(self, func, dropna=..., *args, **kwargs):
        """
        Return a copy of a DataFrame excluding filtered elements.

        Elements from groups are filtered if they do not satisfy the
        boolean criterion specified by func.

        Parameters
        ----------
        func : function
            Function to apply to each subframe. Should return True or False.
        dropna : Drop groups that do not pass the filter. True by default;
            If False, groups that evaluate False are filled with NaNs.

        Returns
        -------
        filtered : DataFrame

        Notes
        -----
        Each subframe is endowed the attribute 'name' in case you need to know
        which group you are working on.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
        ...                           'foo', 'bar'],
        ...                    'B' : [1, 2, 3, 4, 5, 6],
        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]})
        >>> grouped = df.groupby('A')
        >>> grouped.filter(lambda x: x['B'].mean() > 3.)
             A  B    C
        1  bar  2  5.0
        3  bar  4  1.0
        5  bar  6  9.0
        """
        ...
    def __getitem__(self, key) -> DataFrameGroupBy | SeriesGroupBy: ...
    def count(self) -> DataFrame:
        """
        Compute count of group, excluding missing values.

        Returns
        -------
        DataFrame
            Count of values within each group.
        """
        ...
    def nunique(self, dropna: bool = ...) -> DataFrame:
        """
        Return DataFrame with counts of unique elements in each position.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        nunique: DataFrame

        Examples
        --------
        >>> df = pd.DataFrame({'id': ['spam', 'egg', 'egg', 'spam',
        ...                           'ham', 'ham'],
        ...                    'value1': [1, 5, 5, 2, 5, 5],
        ...                    'value2': list('abbaxy')})
        >>> df
             id  value1 value2
        0  spam       1      a
        1   egg       5      b
        2   egg       5      b
        3  spam       2      a
        4   ham       5      x
        5   ham       5      y

        >>> df.groupby('id').nunique()
              value1  value2
        id
        egg        1       1
        ham        1       2
        spam       2       1

        Check for rows with the same id but conflicting values:

        >>> df.groupby('id').filter(lambda g: (g.nunique() > 1).any())
             id  value1 value2
        0  spam       1      a
        3  spam       2      a
        4   ham       5      x
        5   ham       5      y
        """
        ...
    @Appender(DataFrame.idxmax.__doc__)
    def idxmax(self, axis=..., skipna: bool = ...): ...
    @Appender(DataFrame.idxmin.__doc__)
    def idxmin(self, axis=..., skipna: bool = ...): ...
    boxplot = ...
