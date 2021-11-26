import types
from typing import TYPE_CHECKING, Sequence
from pandas import DataFrame
from pandas._typing import IndexLabel
from pandas.core.base import PandasObject
from pandas.util._decorators import Appender, Substitution

if TYPE_CHECKING: ...

def hist_series(
    self,
    by=...,
    ax=...,
    grid: bool = ...,
    xlabelsize: int | None = ...,
    xrot: float | None = ...,
    ylabelsize: int | None = ...,
    yrot: float | None = ...,
    figsize: tuple[int, int] | None = ...,
    bins: int | Sequence[int] = ...,
    backend: str | None = ...,
    legend: bool = ...,
    **kwargs
): ...
def hist_frame(
    data: DataFrame,
    column: IndexLabel = ...,
    by=...,
    grid: bool = ...,
    xlabelsize: int | None = ...,
    xrot: float | None = ...,
    ylabelsize: int | None = ...,
    yrot: float | None = ...,
    ax=...,
    sharex: bool = ...,
    sharey: bool = ...,
    figsize: tuple[int, int] | None = ...,
    layout: tuple[int, int] | None = ...,
    bins: int | Sequence[int] = ...,
    backend: str | None = ...,
    legend: bool = ...,
    **kwargs
): ...

_boxplot_doc = ...
_backend_doc = ...
_bar_or_line_doc = ...

@Substitution(backend="")
@Appender(_boxplot_doc)
def boxplot(
    data,
    column=...,
    by=...,
    ax=...,
    fontsize=...,
    rot=...,
    grid=...,
    figsize=...,
    layout=...,
    return_type=...,
    **kwargs
): ...
@Substitution(backend=_backend_doc)
@Appender(_boxplot_doc)
def boxplot_frame(
    self,
    column=...,
    by=...,
    ax=...,
    fontsize=...,
    rot=...,
    grid=...,
    figsize=...,
    layout=...,
    return_type=...,
    backend=...,
    **kwargs
): ...
def boxplot_frame_groupby(
    grouped,
    subplots=...,
    column=...,
    fontsize=...,
    rot=...,
    grid=...,
    ax=...,
    figsize=...,
    layout=...,
    sharex=...,
    sharey=...,
    backend=...,
    **kwargs
): ...

class PlotAccessor(PandasObject):
    _common_kinds = ...
    _series_kinds = ...
    _dataframe_kinds = ...
    _kind_aliases = ...
    _all_kinds = ...
    def __init__(self, data) -> None: ...
    def __call__(self, *args, **kwargs): ...
    @Appender(
        """
        See Also
        --------
        matplotlib.pyplot.plot : Plot y versus x as lines and/or markers.
        Examples
        --------
        .. plot::
            :context: close-figs
            >>> s = pd.Series([1, 3, 2])
            >>> s.plot.line()
        .. plot::
            :context: close-figs
            The following example shows the populations for some animals
            over the years.
            >>> df = pd.DataFrame({
            ...    'pig': [20, 18, 489, 675, 1776],
            ...    'horse': [4, 25, 281, 600, 1900]
            ...    }, index=[1990, 1997, 2003, 2009, 2014])
            >>> lines = df.plot.line()
        .. plot::
           :context: close-figs
           An example with subplots, so an array of axes is returned.
           >>> axes = df.plot.line(subplots=True)
           >>> type(axes)
           <class 'numpy.ndarray'>
        .. plot::
           :context: close-figs
           Let's repeat the same example, but specifying colors for
           each column (in this case, for each animal).
           >>> axes = df.plot.line(
           ...     subplots=True, color={"pig": "pink", "horse": "#742802"}
           ... )
        .. plot::
            :context: close-figs
            The following example shows the relationship between both
            populations.
            >>> lines = df.plot.line(x='pig', y='horse')
        """
    )
    @Substitution(kind="line")
    @Appender(_bar_or_line_doc)
    def line(self, x=..., y=..., **kwargs): ...
    @Appender(
        """
        See Also
        --------
        DataFrame.plot.barh : Horizontal bar plot.
        DataFrame.plot : Make plots of a DataFrame.
        matplotlib.pyplot.bar : Make a bar plot with matplotlib.
        Examples
        --------
        Basic plot.
        .. plot::
            :context: close-figs
            >>> df = pd.DataFrame({'lab':['A', 'B', 'C'], 'val':[10, 30, 20]})
            >>> ax = df.plot.bar(x='lab', y='val', rot=0)
        Plot a whole dataframe to a bar plot. Each column is assigned a
        distinct color, and each row is nested in a group along the
        horizontal axis.
        .. plot::
            :context: close-figs
            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.bar(rot=0)
        Plot stacked bar charts for the DataFrame
        .. plot::
            :context: close-figs
            >>> ax = df.plot.bar(stacked=True)
        Instead of nesting, the figure can be split by column with
        ``subplots=True``. In this case, a :class:`numpy.ndarray` of
        :class:`matplotlib.axes.Axes` are returned.
        .. plot::
            :context: close-figs
            >>> axes = df.plot.bar(rot=0, subplots=True)
            >>> axes[1].legend(loc=2)  # doctest: +SKIP
        If you don't like the default colours, you can specify how you'd
        like each column to be colored.
        .. plot::
            :context: close-figs
            >>> axes = df.plot.bar(
            ...     rot=0, subplots=True, color={"speed": "red", "lifespan": "green"}
            ... )
            >>> axes[1].legend(loc=2)  # doctest: +SKIP
        Plot a single column.
        .. plot::
            :context: close-figs
            >>> ax = df.plot.bar(y='speed', rot=0)
        Plot only selected categories for the DataFrame.
        .. plot::
            :context: close-figs
            >>> ax = df.plot.bar(x='lifespan', rot=0)
    """
    )
    @Substitution(kind="bar")
    @Appender(_bar_or_line_doc)
    def bar(self, x=..., y=..., **kwargs): ...
    @Appender(
        """
        See Also
        --------
        DataFrame.plot.bar: Vertical bar plot.
        DataFrame.plot : Make plots of DataFrame using matplotlib.
        matplotlib.axes.Axes.bar : Plot a vertical bar plot using matplotlib.
        Examples
        --------
        Basic example
        .. plot::
            :context: close-figs
            >>> df = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
            >>> ax = df.plot.barh(x='lab', y='val')
        Plot a whole DataFrame to a horizontal bar plot
        .. plot::
            :context: close-figs
            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh()
        Plot stacked barh charts for the DataFrame
        .. plot::
            :context: close-figs
            >>> ax = df.plot.barh(stacked=True)
        We can specify colors for each column
        .. plot::
            :context: close-figs
            >>> ax = df.plot.barh(color={"speed": "red", "lifespan": "green"})
        Plot a column of the DataFrame to a horizontal bar plot
        .. plot::
            :context: close-figs
            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh(y='speed')
        Plot DataFrame versus the desired column
        .. plot::
            :context: close-figs
            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = pd.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh(x='lifespan')
    """
    )
    @Substitution(kind="bar")
    @Appender(_bar_or_line_doc)
    def barh(self, x=..., y=..., **kwargs): ...
    def box(self, by=..., **kwargs): ...
    def hist(self, by=..., bins=..., **kwargs): ...
    def kde(self, bw_method=..., ind=..., **kwargs): ...
    density = ...
    def area(self, x=..., y=..., **kwargs): ...
    def pie(self, **kwargs): ...
    def scatter(self, x, y, s=..., c=..., **kwargs): ...
    def hexbin(self, x, y, C=..., reduce_C_function=..., gridsize=..., **kwargs): ...

_backends: dict[str, types.ModuleType] = ...
