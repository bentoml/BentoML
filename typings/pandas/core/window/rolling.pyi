from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Hashable
from pandas._typing import Axis, FrameOrSeries, FrameOrSeriesUnion
from pandas.core.base import SelectionMixin
from pandas.core.groupby.ops import BaseGrouper
from pandas.core.indexes.api import Index
from pandas.core.window.doc import (
    _shared_docs,
    args_compat,
    create_section_header,
    kwargs_compat,
    kwargs_scipy,
    numba_notes,
    template_header,
    template_returns,
    template_see_also,
    window_agg_numba_parameters,
    window_apply_parameters,
)
from pandas.util._decorators import doc

if TYPE_CHECKING: ...

class BaseWindow(SelectionMixin):
    _attributes: list[str] = ...
    exclusions: frozenset[Hashable] = ...
    _on: Index
    def __init__(
        self,
        obj: FrameOrSeries,
        window=...,
        min_periods: int | None = ...,
        center: bool = ...,
        win_type: str | None = ...,
        axis: Axis = ...,
        on: str | Index | None = ...,
        closed: str | None = ...,
        method: str = ...,
        *,
        selection=...
    ) -> None: ...
    @property
    def win_type(self): ...
    @property
    def is_datetimelike(self) -> bool: ...
    def validate(self) -> None: ...
    def __getattr__(self, attr: str): ...
    def __repr__(self) -> str: ...
    def __iter__(self): ...
    def aggregate(self, func, *args, **kwargs): ...
    agg = ...

class BaseWindowGroupby(BaseWindow):
    _grouper: BaseGrouper
    _as_index: bool
    _attributes = ...
    def __init__(
        self,
        obj: FrameOrSeries,
        *args,
        _grouper: BaseGrouper,
        _as_index: bool = ...,
        **kwargs
    ) -> None: ...

class Window(BaseWindow):
    _attributes = ...
    def validate(self): ...
    @doc(
        _shared_docs["aggregate"],
        see_also=dedent(
            """
        See Also
        --------
        pandas.DataFrame.aggregate : Similar DataFrame method.
        pandas.Series.aggregate : Similar Series method.
        """
        ),
        examples=dedent(
            """
        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9
        >>> df.rolling(2, win_type="boxcar").agg("mean")
             A    B    C
        0  NaN  NaN  NaN
        1  1.5  4.5  7.5
        2  2.5  5.5  8.5
        """
        ),
        klass="Series/DataFrame",
        axis="",
    )
    def aggregate(self, func, *args, **kwargs): ...
    agg = ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_scipy,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="rolling",
        aggregation_description="weighted window sum",
        agg_method="sum",
    )
    def sum(self, *args, **kwargs): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_scipy,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="rolling",
        aggregation_description="weighted window mean",
        agg_method="mean",
    )
    def mean(self, *args, **kwargs): ...
    @doc(
        template_header,
        ".. versionadded:: 1.0.0 \n\n",
        create_section_header("Parameters"),
        kwargs_scipy,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="rolling",
        aggregation_description="weighted window variance",
        agg_method="var",
    )
    def var(self, ddof: int = ..., *args, **kwargs): ...
    @doc(
        template_header,
        ".. versionadded:: 1.0.0 \n\n",
        create_section_header("Parameters"),
        kwargs_scipy,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="rolling",
        aggregation_description="weighted window standard deviation",
        agg_method="std",
    )
    def std(self, ddof: int = ..., *args, **kwargs): ...

class RollingAndExpandingMixin(BaseWindow):
    def count(self): ...
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = ...,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        args: tuple[Any, ...] | None = ...,
        kwargs: dict[str, Any] | None = ...,
    ): ...
    def sum(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs
    ): ...
    def max(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs
    ): ...
    def min(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs
    ): ...
    def mean(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs
    ): ...
    def median(
        self,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs
    ): ...
    def std(self, ddof: int = ..., *args, **kwargs): ...
    def var(self, ddof: int = ..., *args, **kwargs): ...
    def skew(self, **kwargs): ...
    def sem(self, ddof: int = ..., *args, **kwargs): ...
    def kurt(self, **kwargs): ...
    def quantile(self, quantile: float, interpolation: str = ..., **kwargs): ...
    def cov(
        self,
        other: FrameOrSeriesUnion | None = ...,
        pairwise: bool | None = ...,
        ddof: int = ...,
        **kwargs
    ): ...
    def corr(
        self,
        other: FrameOrSeriesUnion | None = ...,
        pairwise: bool | None = ...,
        ddof: int = ...,
        **kwargs
    ): ...

class Rolling(RollingAndExpandingMixin):
    _attributes = ...
    def validate(self): ...
    @doc(
        _shared_docs["aggregate"],
        see_also=dedent(
            """
        See Also
        --------
        pandas.Series.rolling : Calling object with Series data.
        pandas.DataFrame.rolling : Calling object with DataFrame data.
        """
        ),
        examples=dedent(
            """
        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9
        >>> df.rolling(2).sum()
             A     B     C
        0  NaN   NaN   NaN
        1  3.0   9.0  15.0
        2  5.0  11.0  17.0
        >>> df.rolling(2).agg({"A": "sum", "B": "min"})
             A    B
        0  NaN  NaN
        1  3.0  4.0
        2  5.0  5.0
        """
        ),
        klass="Series/Dataframe",
        axis="",
    )
    def aggregate(self, func, *args, **kwargs): ...
    agg = ...
    @doc(
        template_header,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([2, 3, np.nan, 10])
        >>> s.rolling(2).count()
        0    1.0
        1    2.0
        2    1.0
        3    1.0
        dtype: float64
        >>> s.rolling(3).count()
        0    1.0
        1    2.0
        2    2.0
        3    2.0
        dtype: float64
        >>> s.rolling(4).count()
        0    1.0
        1    2.0
        2    2.0
        3    3.0
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="count of non NaN observations",
        agg_method="count",
    )
    def count(self): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        window_apply_parameters,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="rolling",
        aggregation_description="custom aggregation function",
        agg_method="apply",
    )
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = ...,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        args: tuple[Any, ...] | None = ...,
        kwargs: dict[str, Any] | None = ...,
    ): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        args_compat,
        window_agg_numba_parameters,
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> s
        0    1
        1    2
        2    3
        3    4
        4    5
        dtype: int64
        >>> s.rolling(3).sum()
        0     NaN
        1     NaN
        2     6.0
        3     9.0
        4    12.0
        dtype: float64
        >>> s.rolling(3, center=True).sum()
        0     NaN
        1     6.0
        2     9.0
        3    12.0
        4     NaN
        dtype: float64
        For DataFrame, each sum is computed column-wise.
        >>> df = pd.DataFrame({{"A": s, "B": s ** 2}})
        >>> df
           A   B
        0  1   1
        1  2   4
        2  3   9
        3  4  16
        4  5  25
        >>> df.rolling(3).sum()
              A     B
        0   NaN   NaN
        1   NaN   NaN
        2   6.0  14.0
        3   9.0  29.0
        4  12.0  50.0
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="sum",
        agg_method="sum",
    )
    def sum(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs
    ): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        args_compat,
        window_agg_numba_parameters,
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes[:-1],
        window_method="rolling",
        aggregation_description="maximum",
        agg_method="max",
    )
    def max(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs
    ): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        args_compat,
        window_agg_numba_parameters,
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """
        Performing a rolling minimum with a window size of 3.
        >>> s = pd.Series([4, 3, 5, 2, 6])
        >>> s.rolling(3).min()
        0    NaN
        1    NaN
        2    3.0
        3    2.0
        4    2.0
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="minimum",
        agg_method="min",
    )
    def min(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs
    ): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        args_compat,
        window_agg_numba_parameters,
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """
        The below examples will show rolling mean calculations with window sizes of
        two and three, respectively.
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.rolling(2).mean()
        0    NaN
        1    1.5
        2    2.5
        3    3.5
        dtype: float64
        >>> s.rolling(3).mean()
        0    NaN
        1    NaN
        2    2.0
        3    3.0
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="mean",
        agg_method="mean",
    )
    def mean(
        self,
        *args,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs
    ): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        window_agg_numba_parameters,
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """
        Compute the rolling median of a series with a window size of 3.
        >>> s = pd.Series([0, 1, 2, 3, 4])
        >>> s.rolling(3).median()
        0    NaN
        1    NaN
        2    1.0
        3    2.0
        4    3.0
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="median",
        agg_method="median",
    )
    def median(
        self,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        **kwargs
    ): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),
        args_compat,
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        "numpy.std : Equivalent method for NumPy array.\n",
        template_see_also,
        create_section_header("Notes"),
        dedent(
            """
        The default ``ddof`` of 1 used in :meth:`Series.std` is different
        than the default ``ddof`` of 0 in :func:`numpy.std`.
        A minimum of one period is required for the rolling calculation.
        The implementation is susceptible to floating point imprecision as
        shown in the example below.\n
        """
        ).replace("\n", "", 1),
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.rolling(3).std()
        0             NaN
        1             NaN
        2    5.773503e-01
        3    1.000000e+00
        4    1.000000e+00
        5    1.154701e+00
        6    2.580957e-08
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="standard deviation",
        agg_method="std",
    )
    def std(self, ddof: int = ..., *args, **kwargs): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),
        args_compat,
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        "numpy.var : Equivalent method for NumPy array.\n",
        template_see_also,
        create_section_header("Notes"),
        dedent(
            """
        The default ``ddof`` of 1 used in :meth:`Series.var` is different
        than the default ``ddof`` of 0 in :func:`numpy.var`.
        A minimum of one period is required for the rolling calculation.
        The implementation is susceptible to floating point imprecision as
        shown in the example below.\n
        """
        ).replace("\n", "", 1),
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.rolling(3).var()
        0             NaN
        1             NaN
        2    3.333333e-01
        3    1.000000e+00
        4    1.000000e+00
        5    1.333333e+00
        6    6.661338e-16
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="variance",
        agg_method="var",
    )
    def var(self, ddof: int = ..., *args, **kwargs): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        "scipy.stats.skew : Third moment of a probability density.\n",
        template_see_also,
        create_section_header("Notes"),
        "A minimum of three periods is required for the rolling calculation.\n",
        window_method="rolling",
        aggregation_description="unbiased skewness",
        agg_method="skew",
    )
    def skew(self, **kwargs): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),
        args_compat,
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        "A minimum of one period is required for the calculation.\n\n",
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([0, 1, 2, 3])
        >>> s.rolling(2, min_periods=1).sem()
        0         NaN
        1    0.707107
        2    0.707107
        3    0.707107
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="standard error of mean",
        agg_method="sem",
    )
    def sem(self, ddof: int = ..., *args, **kwargs): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        "scipy.stats.kurtosis : Reference SciPy method.\n",
        template_see_also,
        create_section_header("Notes"),
        "A minimum of four periods is required for the calculation.\n\n",
        create_section_header("Examples"),
        dedent(
            """
        The example below will show a rolling calculation with a window size of
        four matching the equivalent function call using `scipy.stats`.
        >>> arr = [1, 2, 3, 4, 999]
        >>> import scipy.stats
        >>> print(f"{{scipy.stats.kurtosis(arr[:-1], bias=False):.6f}}")
        -1.200000
        >>> print(f"{{scipy.stats.kurtosis(arr[1:], bias=False):.6f}}")
        3.999946
        >>> s = pd.Series(arr)
        >>> s.rolling(4).kurt()
        0         NaN
        1         NaN
        2         NaN
        3   -1.200000
        4    3.999946
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="Fisher's definition of kurtosis without bias",
        agg_method="kurt",
    )
    def kurt(self, **kwargs): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        quantile : float
            Quantile to compute. 0 <= quantile <= 1.
        interpolation : {{'linear', 'lower', 'higher', 'midpoint', 'nearest'}}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:
                * linear: `i + (j - i) * fraction`, where `fraction` is the
                  fractional part of the index surrounded by `i` and `j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.
        """
        ).replace("\n", "", 1),
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.rolling(2).quantile(.4, interpolation='lower')
        0    NaN
        1    1.0
        2    2.0
        3    3.0
        dtype: float64
        >>> s.rolling(2).quantile(.4, interpolation='midpoint')
        0    NaN
        1    1.5
        2    2.5
        3    3.5
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="quantile",
        agg_method="quantile",
    )
    def quantile(self, quantile: float, interpolation: str = ..., **kwargs): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndexed DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="rolling",
        aggregation_description="sample covariance",
        agg_method="cov",
    )
    def cov(
        self,
        other: FrameOrSeriesUnion | None = ...,
        pairwise: bool | None = ...,
        ddof: int = ...,
        **kwargs
    ): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndexed DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        dedent(
            """
        cov : Similar method to calculate covariance.
        numpy.corrcoef : NumPy Pearson's correlation calculation.
        """
        ).replace("\n", "", 1),
        template_see_also,
        create_section_header("Notes"),
        dedent(
            """
        This function uses Pearson's definition of correlation
        (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
        When `other` is not specified, the output will be self correlation (e.g.
        all 1's), except for :class:`~pandas.DataFrame` inputs with `pairwise`
        set to `True`.
        Function will return ``NaN`` for correlations of equal valued sequences;
        this is the result of a 0/0 division error.
        When `pairwise` is set to `False`, only matching columns between `self` and
        `other` will be used.
        When `pairwise` is set to `True`, the output will be a MultiIndex DataFrame
        with the original index on the first level, and the `other` DataFrame
        columns on the second level.
        In the case of missing elements, only complete pairwise observations
        will be used.\n
        """
        ).replace("\n", "", 1),
        create_section_header("Examples"),
        dedent(
            """
        The below example shows a rolling calculation with a window size of
        four matching the equivalent function call using :meth:`numpy.corrcoef`.
        >>> v1 = [3, 3, 3, 5, 8]
        >>> v2 = [3, 4, 4, 4, 8]
        >>> # numpy returns a 2X2 array, the correlation coefficient
        >>> # is the number at entry [0][1]
        >>> print(f"{{np.corrcoef(v1[:-1], v2[:-1])[0][1]:.6f}}")
        0.333333
        >>> print(f"{{np.corrcoef(v1[1:], v2[1:])[0][1]:.6f}}")
        0.916949
        >>> s1 = pd.Series(v1)
        >>> s2 = pd.Series(v2)
        >>> s1.rolling(4).corr(s2)
        0         NaN
        1         NaN
        2         NaN
        3    0.333333
        4    0.916949
        dtype: float64
        The below example shows a similar rolling calculation on a
        DataFrame using the pairwise option.
        >>> matrix = np.array([[51., 35.], [49., 30.], [47., 32.],        [46., 31.], [50., 36.]])
        >>> print(np.corrcoef(matrix[:-1,0], matrix[:-1,1]).round(7))
        [[1.         0.6263001]
         [0.6263001  1.       ]]
        >>> print(np.corrcoef(matrix[1:,0], matrix[1:,1]).round(7))
        [[1.         0.5553681]
         [0.5553681  1.        ]]
        >>> df = pd.DataFrame(matrix, columns=['X','Y'])
        >>> df
              X     Y
        0  51.0  35.0
        1  49.0  30.0
        2  47.0  32.0
        3  46.0  31.0
        4  50.0  36.0
        >>> df.rolling(4).corr(pairwise=True)
                    X         Y
        0 X       NaN       NaN
          Y       NaN       NaN
        1 X       NaN       NaN
          Y       NaN       NaN
        2 X       NaN       NaN
          Y       NaN       NaN
        3 X  1.000000  0.626300
          Y  0.626300  1.000000
        4 X  1.000000  0.555368
          Y  0.555368  1.000000
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="correlation",
        agg_method="corr",
    )
    def corr(
        self,
        other: FrameOrSeriesUnion | None = ...,
        pairwise: bool | None = ...,
        ddof: int = ...,
        **kwargs
    ): ...

class RollingGroupby(BaseWindowGroupby, Rolling):
    _attributes = ...
