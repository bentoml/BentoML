from typing import TYPE_CHECKING

from pandas import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.util._decorators import Appender, deprecate_kwarg

if TYPE_CHECKING: ...

@Appender(_shared_docs["melt"] % {"caller": "pd.melt(df, ", "other": "DataFrame.melt"})
def melt(
    frame: DataFrame,
    id_vars=...,
    value_vars=...,
    var_name=...,
    value_name=...,
    col_level=...,
    ignore_index: bool = ...,
) -> DataFrame: ...
@deprecate_kwarg(old_arg_name="label", new_arg_name=None)
def lreshape(data: DataFrame, groups, dropna: bool = ..., label=...) -> DataFrame:
    """
    Reshape wide-format data to long. Generalized inverse of DataFrame.pivot.

    Accepts a dictionary, ``groups``, in which each key is a new column name
    and each value is a list of old column names that will be "melted" under
    the new column name as part of the reshape.

    Parameters
    ----------
    data : DataFrame
        The wide-format DataFrame.
    groups : dict
        {new_name : list_of_columns}.
    dropna : bool, default True
        Do not include columns whose entries are all NaN.
    label : None
        Not used.

        .. deprecated:: 1.0.0

    Returns
    -------
    DataFrame
        Reshaped DataFrame.

    See Also
    --------
    melt : Unpivot a DataFrame from wide to long format, optionally leaving
        identifiers set.
    pivot : Create a spreadsheet-style pivot table as a DataFrame.
    DataFrame.pivot : Pivot without aggregation that can handle
        non-numeric data.
    DataFrame.pivot_table : Generalization of pivot that can handle
        duplicate values for one index/column pair.
    DataFrame.unstack : Pivot based on the index values instead of a
        column.
    wide_to_long : Wide panel to long format. Less flexible but more
        user-friendly than melt.

    Examples
    --------
    >>> data = pd.DataFrame({'hr1': [514, 573], 'hr2': [545, 526],
    ...                      'team': ['Red Sox', 'Yankees'],
    ...                      'year1': [2007, 2007], 'year2': [2008, 2008]})
    >>> data
       hr1  hr2     team  year1  year2
    0  514  545  Red Sox   2007   2008
    1  573  526  Yankees   2007   2008

    >>> pd.lreshape(data, {'year': ['year1', 'year2'], 'hr': ['hr1', 'hr2']})
          team  year   hr
    0  Red Sox  2007  514
    1  Yankees  2007  573
    2  Red Sox  2008  545
    3  Yankees  2008  526
    """
    ...

def wide_to_long(
    df: DataFrame, stubnames, i, j, sep: str = ..., suffix: str = ...
) -> DataFrame:
    r"""
    Wide panel to long format. Less flexible but more user-friendly than melt.

    With stubnames ['A', 'B'], this function expects to find one or more
    group of columns with format
    A-suffix1, A-suffix2,..., B-suffix1, B-suffix2,...
    You specify what you want to call this suffix in the resulting long format
    with `j` (for example `j='year'`)

    Each row of these wide variables are assumed to be uniquely identified by
    `i` (can be a single column name or a list of column names)

    All remaining variables in the data frame are left intact.

    Parameters
    ----------
    df : DataFrame
        The wide-format DataFrame.
    stubnames : str or list-like
        The stub name(s). The wide format variables are assumed to
        start with the stub names.
    i : str or list-like
        Column(s) to use as id variable(s).
    j : str
        The name of the sub-observation variable. What you wish to name your
        suffix in the long format.
    sep : str, default ""
        A character indicating the separation of the variable names
        in the wide format, to be stripped from the names in the long format.
        For example, if your column names are A-suffix1, A-suffix2, you
        can strip the hyphen by specifying `sep='-'`.
    suffix : str, default '\\d+'
        A regular expression capturing the wanted suffixes. '\\d+' captures
        numeric suffixes. Suffixes with no numbers could be specified with the
        negated character class '\\D+'. You can also further disambiguate
        suffixes, for example, if your wide variables are of the form A-one,
        B-two,.., and you have an unrelated column A-rating, you can ignore the
        last one by specifying `suffix='(!?one|two)'`. When all suffixes are
        numeric, they are cast to int64/float64.

    Returns
    -------
    DataFrame
        A DataFrame that contains each stub name as a variable, with new index
        (i, j).

    See Also
    --------
    melt : Unpivot a DataFrame from wide to long format, optionally leaving
        identifiers set.
    pivot : Create a spreadsheet-style pivot table as a DataFrame.
    DataFrame.pivot : Pivot without aggregation that can handle
        non-numeric data.
    DataFrame.pivot_table : Generalization of pivot that can handle
        duplicate values for one index/column pair.
    DataFrame.unstack : Pivot based on the index values instead of a
        column.

    Notes
    -----
    All extra variables are left untouched. This simply uses
    `pandas.melt` under the hood, but is hard-coded to "do the right thing"
    in a typical case.

    Examples
    --------
    >>> np.random.seed(123)
    >>> df = pd.DataFrame({"A1970" : {0 : "a", 1 : "b", 2 : "c"},
    ...                    "A1980" : {0 : "d", 1 : "e", 2 : "f"},
    ...                    "B1970" : {0 : 2.5, 1 : 1.2, 2 : .7},
    ...                    "B1980" : {0 : 3.2, 1 : 1.3, 2 : .1},
    ...                    "X"     : dict(zip(range(3), np.random.randn(3)))
    ...                   })
    >>> df["id"] = df.index
    >>> df
      A1970 A1980  B1970  B1980         X  id
    0     a     d    2.5    3.2 -1.085631   0
    1     b     e    1.2    1.3  0.997345   1
    2     c     f    0.7    0.1  0.282978   2
    >>> pd.wide_to_long(df, ["A", "B"], i="id", j="year")
    ... # doctest: +NORMALIZE_WHITESPACE
                    X  A    B
    id year
    0  1970 -1.085631  a  2.5
    1  1970  0.997345  b  1.2
    2  1970  0.282978  c  0.7
    0  1980 -1.085631  d  3.2
    1  1980  0.997345  e  1.3
    2  1980  0.282978  f  0.1

    With multiple id columns

    >>> df = pd.DataFrame({
    ...     'famid': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...     'birth': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    ...     'ht1': [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
    ...     'ht2': [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9]
    ... })
    >>> df
       famid  birth  ht1  ht2
    0      1      1  2.8  3.4
    1      1      2  2.9  3.8
    2      1      3  2.2  2.9
    3      2      1  2.0  3.2
    4      2      2  1.8  2.8
    5      2      3  1.9  2.4
    6      3      1  2.2  3.3
    7      3      2  2.3  3.4
    8      3      3  2.1  2.9
    >>> l = pd.wide_to_long(df, stubnames='ht', i=['famid', 'birth'], j='age')
    >>> l
    ... # doctest: +NORMALIZE_WHITESPACE
                      ht
    famid birth age
    1     1     1    2.8
                2    3.4
          2     1    2.9
                2    3.8
          3     1    2.2
                2    2.9
    2     1     1    2.0
                2    3.2
          2     1    1.8
                2    2.8
          3     1    1.9
                2    2.4
    3     1     1    2.2
                2    3.3
          2     1    2.3
                2    3.4
          3     1    2.1
                2    2.9

    Going from long back to wide just takes some creative use of `unstack`

    >>> w = l.unstack()
    >>> w.columns = w.columns.map('{0[0]}{0[1]}'.format)
    >>> w.reset_index()
       famid  birth  ht1  ht2
    0      1      1  2.8  3.4
    1      1      2  2.9  3.8
    2      1      3  2.2  2.9
    3      2      1  2.0  3.2
    4      2      2  1.8  2.8
    5      2      3  1.9  2.4
    6      3      1  2.2  3.3
    7      3      2  2.3  3.4
    8      3      3  2.1  2.9

    Less wieldy column names are also handled

    >>> np.random.seed(0)
    >>> df = pd.DataFrame({'A(weekly)-2010': np.random.rand(3),
    ...                    'A(weekly)-2011': np.random.rand(3),
    ...                    'B(weekly)-2010': np.random.rand(3),
    ...                    'B(weekly)-2011': np.random.rand(3),
    ...                    'X' : np.random.randint(3, size=3)})
    >>> df['id'] = df.index
    >>> df # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
       A(weekly)-2010  A(weekly)-2011  B(weekly)-2010  B(weekly)-2011  X  id
    0        0.548814        0.544883        0.437587        0.383442  0   0
    1        0.715189        0.423655        0.891773        0.791725  1   1
    2        0.602763        0.645894        0.963663        0.528895  1   2

    >>> pd.wide_to_long(df, ['A(weekly)', 'B(weekly)'], i='id',
    ...                 j='year', sep='-')
    ... # doctest: +NORMALIZE_WHITESPACE
             X  A(weekly)  B(weekly)
    id year
    0  2010  0   0.548814   0.437587
    1  2010  1   0.715189   0.891773
    2  2010  1   0.602763   0.963663
    0  2011  0   0.544883   0.383442
    1  2011  1   0.423655   0.791725
    2  2011  1   0.645894   0.528895

    If we have many columns, we could also use a regex to find our
    stubnames and pass that list on to wide_to_long

    >>> stubnames = sorted(
    ...     set([match[0] for match in df.columns.str.findall(
    ...         r'[A-B]\(.*\)').values if match != []])
    ... )
    >>> list(stubnames)
    ['A(weekly)', 'B(weekly)']

    All of the above examples have integers as suffixes. It is possible to
    have non-integers as suffixes.

    >>> df = pd.DataFrame({
    ...     'famid': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...     'birth': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    ...     'ht_one': [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
    ...     'ht_two': [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9]
    ... })
    >>> df
       famid  birth  ht_one  ht_two
    0      1      1     2.8     3.4
    1      1      2     2.9     3.8
    2      1      3     2.2     2.9
    3      2      1     2.0     3.2
    4      2      2     1.8     2.8
    5      2      3     1.9     2.4
    6      3      1     2.2     3.3
    7      3      2     2.3     3.4
    8      3      3     2.1     2.9

    >>> l = pd.wide_to_long(df, stubnames='ht', i=['famid', 'birth'], j='age',
    ...                     sep='_', suffix=r'\w+')
    >>> l
    ... # doctest: +NORMALIZE_WHITESPACE
                      ht
    famid birth age
    1     1     one  2.8
                two  3.4
          2     one  2.9
                two  3.8
          3     one  2.2
                two  2.9
    2     1     one  2.0
                two  3.2
          2     one  1.8
                two  2.8
          3     one  1.9
                two  2.4
    3     1     one  2.2
                two  3.3
          2     one  2.3
                two  3.4
          3     one  2.1
                two  2.9
    """
    ...
