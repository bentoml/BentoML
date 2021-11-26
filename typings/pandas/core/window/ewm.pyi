from textwrap import dedent
import numpy as np
from pandas._typing import (
    Axis,
    FrameOrSeries,
    FrameOrSeriesUnion,
    TimedeltaConvertibleTypes,
)
from pandas.core.window.doc import (
    _shared_docs,
    args_compat,
    create_section_header,
    kwargs_compat,
    numba_notes,
    template_header,
    template_returns,
    template_see_also,
    window_agg_numba_parameters,
)
from pandas.core.window.rolling import BaseWindow, BaseWindowGroupby
from pandas.util._decorators import doc

def get_center_of_mass(
    comass: float | None,
    span: float | None,
    halflife: float | None,
    alpha: float | None,
) -> float: ...

class ExponentialMovingWindow(BaseWindow):
    _attributes = ...
    def __init__(
        self,
        obj: FrameOrSeries,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | TimedeltaConvertibleTypes | None = ...,
        alpha: float | None = ...,
        min_periods: int | None = ...,
        adjust: bool = ...,
        ignore_na: bool = ...,
        axis: Axis = ...,
        times: str | np.ndarray | FrameOrSeries | None = ...,
        *,
        selection=...
    ) -> None: ...
    def online(self, engine=..., engine_kwargs=...): ...
    @doc(
        _shared_docs["aggregate"],
        see_also=dedent(
            """
        See Also
        --------
        pandas.DataFrame.rolling.aggregate
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
        >>> df.ewm(alpha=0.5).mean()
                  A         B         C
        0  1.000000  4.000000  7.000000
        1  1.666667  4.666667  7.666667
        2  2.428571  5.428571  8.428571
        """
        ),
        klass="Series/Dataframe",
        axis="",
    )
    def aggregate(self, func, *args, **kwargs): ...
    agg = ...
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
        numba_notes.replace("\n", "", 1),
        window_method="ewm",
        aggregation_description="(exponential weighted moment) mean",
        agg_method="mean",
    )
    def mean(self, *args, engine=..., engine_kwargs=..., **kwargs): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        bias : bool, default False
            Use a standard estimation bias correction.
        """
        ).replace("\n", "", 1),
        args_compat,
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="ewm",
        aggregation_description="(exponential weighted moment) standard deviation",
        agg_method="std",
    )
    def std(self, bias: bool = ..., *args, **kwargs): ...
    def vol(self, bias: bool = ..., *args, **kwargs): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        bias : bool, default False
            Use a standard estimation bias correction.
        """
        ).replace("\n", "", 1),
        args_compat,
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="ewm",
        aggregation_description="(exponential weighted moment) variance",
        agg_method="var",
    )
    def var(self, bias: bool = ..., *args, **kwargs): ...
    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        other : Series or DataFrame , optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndex DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        bias : bool, default False
            Use a standard estimation bias correction.
        """
        ).replace("\n", "", 1),
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="ewm",
        aggregation_description="(exponential weighted moment) sample covariance",
        agg_method="cov",
    )
    def cov(
        self,
        other: FrameOrSeriesUnion | None = ...,
        pairwise: bool | None = ...,
        bias: bool = ...,
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
            output will be a MultiIndex DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        """
        ).replace("\n", "", 1),
        kwargs_compat,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also[:-1],
        window_method="ewm",
        aggregation_description="(exponential weighted moment) sample correlation",
        agg_method="corr",
    )
    def corr(
        self,
        other: FrameOrSeriesUnion | None = ...,
        pairwise: bool | None = ...,
        **kwargs
    ): ...

class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    _attributes = ...
    def __init__(self, obj, *args, _grouper=..., **kwargs) -> None: ...

class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    def __init__(
        self,
        obj: FrameOrSeries,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | TimedeltaConvertibleTypes | None = ...,
        alpha: float | None = ...,
        min_periods: int | None = ...,
        adjust: bool = ...,
        ignore_na: bool = ...,
        axis: Axis = ...,
        times: str | np.ndarray | FrameOrSeries | None = ...,
        engine: str = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        *,
        selection=...
    ) -> None: ...
    def reset(self): ...
    def aggregate(self, func, *args, **kwargs): ...
    def std(self, bias: bool = ..., *args, **kwargs): ...
    def corr(
        self,
        other: FrameOrSeriesUnion | None = ...,
        pairwise: bool | None = ...,
        **kwargs
    ): ...
    def cov(
        self,
        other: FrameOrSeriesUnion | None = ...,
        pairwise: bool | None = ...,
        bias: bool = ...,
        **kwargs
    ): ...
    def var(self, bias: bool = ..., *args, **kwargs): ...
