import weakref
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    AnyStr,
    Callable,
    Hashable,
    Literal,
    Mapping,
    Sequence,
    overload,
)
import numpy as np
from pandas._libs.tslibs import BaseOffset
from pandas._typing import (
    Axis,
    CompressionOptions,
    DtypeArg,
    FilePathOrBuffer,
    FrameOrSeries,
    IndexKeyFunc,
    IndexLabel,
    JSONSerializable,
    Level,
    Manager,
    NpDtype,
    Renamer,
    StorageOptions,
    T,
    TimedeltaConvertibleTypes,
    TimestampConvertibleTypes,
    ValueKeyFunc,
    final,
)
from pandas.core import indexing
from pandas.core.base import PandasObject
from pandas.core.dtypes.missing import isna, notna
from pandas.core.flags import Flags
from pandas.core.indexes.api import Index
from pandas.core.resample import Resampler
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.core.window import Expanding, ExponentialMovingWindow, Rolling
from pandas.core.window.indexers import BaseIndexer
from pandas.io.formats import format as fmt
from pandas.util._decorators import doc, rewrite_axis_style_signature

if TYPE_CHECKING: ...
_shared_docs = ...
_shared_doc_kwargs = ...
bool_t = bool

class NDFrame(PandasObject, indexing.IndexingMixin):
    _internal_names: list[str] = ...
    _internal_names_set: set[str] = ...
    _accessors: set[str] = ...
    _hidden_attrs: frozenset[str] = ...
    _metadata: list[str] = ...
    _is_copy: weakref.ReferenceType[NDFrame] | None = ...
    _mgr: Manager
    _attrs: dict[Hashable, Any]
    _typ: str
    def __init__(
        self,
        data: Manager,
        copy: bool_t = ...,
        attrs: Mapping[Hashable, Any] | None = ...,
    ) -> None: ...
    @property
    def attrs(self) -> dict[Hashable, Any]: ...
    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None: ...
    @final
    @property
    def flags(self) -> Flags: ...
    @final
    def set_flags(
        self: FrameOrSeries,
        *,
        copy: bool_t = ...,
        allows_duplicate_labels: bool_t | None = ...
    ) -> FrameOrSeries: ...
    _stat_axis_number = ...
    _stat_axis_name = ...
    _AXIS_ORDERS: list[str]
    _AXIS_TO_AXIS_NUMBER: dict[Axis, int] = ...
    _AXIS_REVERSED: bool_t
    _info_axis_number: int
    _info_axis_name: str
    _AXIS_LEN: int
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def axes(self) -> list[Index]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @overload
    def set_axis(
        self: FrameOrSeries, labels, axis: Axis = ..., inplace: Literal[False] = ...
    ) -> FrameOrSeries: ...
    @overload
    def set_axis(
        self: FrameOrSeries, labels, axis: Axis, inplace: Literal[True]
    ) -> None: ...
    @overload
    def set_axis(self: FrameOrSeries, labels, *, inplace: Literal[True]) -> None: ...
    @overload
    def set_axis(
        self: FrameOrSeries, labels, axis: Axis = ..., inplace: bool_t = ...
    ) -> FrameOrSeries | None: ...
    def set_axis(self, labels, axis: Axis = ..., inplace: bool_t = ...): ...
    @final
    def swapaxes(self: FrameOrSeries, axis1, axis2, copy=...) -> FrameOrSeries: ...
    @final
    @doc(klass=_shared_doc_kwargs["klass"])
    def droplevel(self: FrameOrSeries, level, axis=...) -> FrameOrSeries: ...
    def pop(self, item: Hashable) -> Series | Any: ...
    @final
    def squeeze(self, axis=...): ...
    def rename(
        self: FrameOrSeries,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        copy: bool_t = ...,
        inplace: bool_t = ...,
        level: Level | None = ...,
        errors: str = ...
    ) -> FrameOrSeries | None: ...
    @rewrite_axis_style_signature("mapper", [("copy", True), ("inplace", False)])
    def rename_axis(self, mapper=..., **kwargs): ...
    @final
    def equals(self, other: object) -> bool_t: ...
    @final
    def __neg__(self): ...
    @final
    def __pos__(self): ...
    @final
    def __invert__(self): ...
    @final
    def __nonzero__(self): ...
    __bool__ = ...
    @final
    def bool(self): ...
    @final
    def __abs__(self: FrameOrSeries) -> FrameOrSeries: ...
    @final
    def __round__(self: FrameOrSeries, decimals: int = ...) -> FrameOrSeries: ...
    __hash__: None
    def __iter__(self): ...
    def keys(self): ...
    def items(self): ...
    @doc(items)
    def iteritems(self): ...
    def __len__(self) -> int: ...
    @final
    def __contains__(self, key) -> bool_t: ...
    @property
    def empty(self) -> bool_t: ...
    __array_priority__ = ...
    def __array__(self, dtype: NpDtype | None = ...) -> np.ndarray: ...
    def __array_wrap__(
        self,
        result: np.ndarray,
        context: tuple[Callable, tuple[Any, ...], int] | None = ...,
    ): ...
    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ): ...
    @final
    def __getstate__(self) -> dict[str, Any]: ...
    @final
    def __setstate__(self, state): ...
    def __repr__(self) -> str: ...
    @final
    @doc(klass="object", storage_options=_shared_docs["storage_options"])
    def to_excel(
        self,
        excel_writer,
        sheet_name: str = ...,
        na_rep: str = ...,
        float_format: str | None = ...,
        columns=...,
        header=...,
        index=...,
        index_label=...,
        startrow=...,
        startcol=...,
        engine=...,
        merge_cells=...,
        encoding=...,
        inf_rep=...,
        verbose=...,
        freeze_panes=...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    @final
    @doc(storage_options=_shared_docs["storage_options"])
    def to_json(
        self,
        path_or_buf: FilePathOrBuffer | None = ...,
        orient: str | None = ...,
        date_format: str | None = ...,
        double_precision: int = ...,
        force_ascii: bool_t = ...,
        date_unit: str = ...,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        lines: bool_t = ...,
        compression: CompressionOptions = ...,
        index: bool_t = ...,
        indent: int | None = ...,
        storage_options: StorageOptions = ...,
    ) -> str | None: ...
    @final
    def to_hdf(
        self,
        path_or_buf,
        key: str,
        mode: str = ...,
        complevel: int | None = ...,
        complib: str | None = ...,
        append: bool_t = ...,
        format: str | None = ...,
        index: bool_t = ...,
        min_itemsize: int | dict[str, int] | None = ...,
        nan_rep=...,
        dropna: bool_t | None = ...,
        data_columns: bool_t | list[str] | None = ...,
        errors: str = ...,
        encoding: str = ...,
    ) -> None: ...
    @final
    def to_sql(
        self,
        name: str,
        con,
        schema=...,
        if_exists: str = ...,
        index: bool_t = ...,
        index_label=...,
        chunksize=...,
        dtype: DtypeArg | None = ...,
        method=...,
    ) -> None: ...
    @final
    @doc(storage_options=_shared_docs["storage_options"])
    def to_pickle(
        self,
        path,
        compression: CompressionOptions = ...,
        protocol: int = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    @final
    def to_clipboard(
        self, excel: bool_t = ..., sep: str | None = ..., **kwargs
    ) -> None: ...
    @final
    def to_xarray(self): ...
    @final
    @doc(returns=fmt.return_docstring)
    def to_latex(
        self,
        buf=...,
        columns=...,
        col_space=...,
        header=...,
        index=...,
        na_rep=...,
        formatters=...,
        float_format=...,
        sparsify=...,
        index_names=...,
        bold_rows=...,
        column_format=...,
        longtable=...,
        escape=...,
        encoding=...,
        decimal=...,
        multicolumn=...,
        multicolumn_format=...,
        multirow=...,
        caption=...,
        label=...,
        position=...,
    ): ...
    @final
    @doc(storage_options=_shared_docs["storage_options"])
    def to_csv(
        self,
        path_or_buf: FilePathOrBuffer[AnyStr] | None = ...,
        sep: str = ...,
        na_rep: str = ...,
        float_format: str | None = ...,
        columns: Sequence[Hashable] | None = ...,
        header: bool_t | list[str] = ...,
        index: bool_t = ...,
        index_label: IndexLabel | None = ...,
        mode: str = ...,
        encoding: str | None = ...,
        compression: CompressionOptions = ...,
        quoting: int | None = ...,
        quotechar: str = ...,
        line_terminator: str | None = ...,
        chunksize: int | None = ...,
        date_format: str | None = ...,
        doublequote: bool_t = ...,
        escapechar: str | None = ...,
        decimal: str = ...,
        errors: str = ...,
        storage_options: StorageOptions = ...,
    ) -> str | None: ...
    def take(
        self: FrameOrSeries, indices, axis=..., is_copy: bool_t | None = ..., **kwargs
    ) -> FrameOrSeries: ...
    @final
    def xs(self, key, axis=..., level=..., drop_level: bool_t = ...): ...
    def __getitem__(self, item): ...
    def __delitem__(self, key) -> None: ...
    @final
    def get(self, key, default=...): ...
    @final
    def reindex_like(
        self: FrameOrSeries,
        other,
        method: str | None = ...,
        copy: bool_t = ...,
        limit=...,
        tolerance=...,
    ) -> FrameOrSeries: ...
    def drop(
        self,
        labels=...,
        axis=...,
        index=...,
        columns=...,
        level=...,
        inplace: bool_t = ...,
        errors: str = ...,
    ): ...
    @final
    def add_prefix(self: FrameOrSeries, prefix: str) -> FrameOrSeries: ...
    @final
    def add_suffix(self: FrameOrSeries, suffix: str) -> FrameOrSeries: ...
    def sort_values(
        self,
        axis=...,
        ascending=...,
        inplace: bool_t = ...,
        kind: str = ...,
        na_position: str = ...,
        ignore_index: bool_t = ...,
        key: ValueKeyFunc = ...,
    ): ...
    def sort_index(
        self,
        axis=...,
        level=...,
        ascending: bool_t | int | Sequence[bool_t | int] = ...,
        inplace: bool_t = ...,
        kind: str = ...,
        na_position: str = ...,
        sort_remaining: bool_t = ...,
        ignore_index: bool_t = ...,
        key: IndexKeyFunc = ...,
    ): ...
    @doc(
        klass=_shared_doc_kwargs["klass"],
        axes=_shared_doc_kwargs["axes"],
        optional_labels="",
        optional_axis="",
    )
    def reindex(self: FrameOrSeries, *args, **kwargs) -> FrameOrSeries: ...
    def filter(
        self: FrameOrSeries,
        items=...,
        like: str | None = ...,
        regex: str | None = ...,
        axis=...,
    ) -> FrameOrSeries: ...
    @final
    def head(self: FrameOrSeries, n: int = ...) -> FrameOrSeries: ...
    @final
    def tail(self: FrameOrSeries, n: int = ...) -> FrameOrSeries: ...
    @final
    def sample(
        self: FrameOrSeries,
        n=...,
        frac: float | None = ...,
        replace: bool_t = ...,
        weights=...,
        random_state=...,
        axis: Axis | None = ...,
        ignore_index: bool_t = ...,
    ) -> FrameOrSeries: ...
    @final
    @doc(klass=_shared_doc_kwargs["klass"])
    def pipe(
        self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs
    ) -> T: ...
    @final
    def __finalize__(
        self: FrameOrSeries, other, method: str | None = ..., **kwargs
    ) -> FrameOrSeries: ...
    def __getattr__(self, name: str): ...
    def __setattr__(self, name: str, value) -> None: ...
    @property
    def values(self) -> np.ndarray: ...
    @property
    def dtypes(self): ...
    def astype(
        self: FrameOrSeries, dtype, copy: bool_t = ..., errors: str = ...
    ) -> FrameOrSeries: ...
    @final
    def copy(self: FrameOrSeries, deep: bool_t = ...) -> FrameOrSeries: ...
    @final
    def __copy__(self: FrameOrSeries, deep: bool_t = ...) -> FrameOrSeries: ...
    @final
    def __deepcopy__(self: FrameOrSeries, memo=...) -> FrameOrSeries: ...
    @final
    def infer_objects(self: FrameOrSeries) -> FrameOrSeries: ...
    @final
    def convert_dtypes(
        self: FrameOrSeries,
        infer_objects: bool_t = ...,
        convert_string: bool_t = ...,
        convert_integer: bool_t = ...,
        convert_boolean: bool_t = ...,
        convert_floating: bool_t = ...,
    ) -> FrameOrSeries: ...
    @doc(**_shared_doc_kwargs)
    def fillna(
        self: FrameOrSeries,
        value=...,
        method=...,
        axis=...,
        inplace: bool_t = ...,
        limit=...,
        downcast=...,
    ) -> FrameOrSeries | None: ...
    @doc(klass=_shared_doc_kwargs["klass"])
    def ffill(
        self: FrameOrSeries,
        axis: None | Axis = ...,
        inplace: bool_t = ...,
        limit: None | int = ...,
        downcast=...,
    ) -> FrameOrSeries | None: ...
    pad = ...
    @doc(klass=_shared_doc_kwargs["klass"])
    def bfill(
        self: FrameOrSeries,
        axis: None | Axis = ...,
        inplace: bool_t = ...,
        limit: None | int = ...,
        downcast=...,
    ) -> FrameOrSeries | None: ...
    backfill = ...
    @doc(
        _shared_docs["replace"],
        klass=_shared_doc_kwargs["klass"],
        inplace=_shared_doc_kwargs["inplace"],
        replace_iloc=_shared_doc_kwargs["replace_iloc"],
    )
    def replace(
        self,
        to_replace=...,
        value=...,
        inplace: bool_t = ...,
        limit: int | None = ...,
        regex=...,
        method=...,
    ): ...
    def interpolate(
        self: FrameOrSeries,
        method: str = ...,
        axis: Axis = ...,
        limit: int | None = ...,
        inplace: bool_t = ...,
        limit_direction: str | None = ...,
        limit_area: str | None = ...,
        downcast: str | None = ...,
        **kwargs
    ) -> FrameOrSeries | None: ...
    @final
    def asof(self, where, subset=...): ...
    @doc(klass=_shared_doc_kwargs["klass"])
    def isna(self: FrameOrSeries) -> FrameOrSeries: ...
    @doc(isna, klass=_shared_doc_kwargs["klass"])
    def isnull(self: FrameOrSeries) -> FrameOrSeries: ...
    @doc(klass=_shared_doc_kwargs["klass"])
    def notna(self: FrameOrSeries) -> FrameOrSeries: ...
    @doc(notna, klass=_shared_doc_kwargs["klass"])
    def notnull(self: FrameOrSeries) -> FrameOrSeries: ...
    def clip(
        self: FrameOrSeries,
        lower=...,
        upper=...,
        axis: Axis | None = ...,
        inplace: bool_t = ...,
        *args,
        **kwargs
    ) -> FrameOrSeries | None: ...
    @doc(**_shared_doc_kwargs)
    def asfreq(
        self: FrameOrSeries,
        freq,
        method=...,
        how: str | None = ...,
        normalize: bool_t = ...,
        fill_value=...,
    ) -> FrameOrSeries: ...
    @final
    def at_time(
        self: FrameOrSeries, time, asof: bool_t = ..., axis=...
    ) -> FrameOrSeries: ...
    @final
    def between_time(
        self: FrameOrSeries,
        start_time,
        end_time,
        include_start: bool_t = ...,
        include_end: bool_t = ...,
        axis=...,
    ) -> FrameOrSeries: ...
    @doc(**_shared_doc_kwargs)
    def resample(
        self,
        rule,
        axis=...,
        closed: str | None = ...,
        label: str | None = ...,
        convention: str = ...,
        kind: str | None = ...,
        loffset=...,
        base: int | None = ...,
        on=...,
        level=...,
        origin: str | TimestampConvertibleTypes = ...,
        offset: TimedeltaConvertibleTypes | None = ...,
    ) -> Resampler: ...
    @final
    def first(self: FrameOrSeries, offset) -> FrameOrSeries: ...
    @final
    def last(self: FrameOrSeries, offset) -> FrameOrSeries: ...
    @final
    def rank(
        self: FrameOrSeries,
        axis=...,
        method: str = ...,
        numeric_only: bool_t | None = ...,
        na_option: str = ...,
        ascending: bool_t = ...,
        pct: bool_t = ...,
    ) -> FrameOrSeries: ...
    @doc(_shared_docs["compare"], klass=_shared_doc_kwargs["klass"])
    def compare(
        self,
        other,
        align_axis: Axis = ...,
        keep_shape: bool_t = ...,
        keep_equal: bool_t = ...,
    ): ...
    @doc(**_shared_doc_kwargs)
    def align(
        self,
        other,
        join=...,
        axis=...,
        level=...,
        copy=...,
        fill_value=...,
        method=...,
        limit=...,
        fill_axis=...,
        broadcast_axis=...,
    ): ...
    @doc(
        klass=_shared_doc_kwargs["klass"],
        cond="True",
        cond_rev="False",
        name="where",
        name_other="mask",
    )
    def where(
        self,
        cond,
        other=...,
        inplace=...,
        axis=...,
        level=...,
        errors=...,
        try_cast=...,
    ): ...
    @final
    @doc(
        where,
        klass=_shared_doc_kwargs["klass"],
        cond="False",
        cond_rev="True",
        name="mask",
        name_other="where",
    )
    def mask(
        self,
        cond,
        other=...,
        inplace=...,
        axis=...,
        level=...,
        errors=...,
        try_cast=...,
    ): ...
    @doc(klass=_shared_doc_kwargs["klass"])
    def shift(
        self: FrameOrSeries, periods=..., freq=..., axis=..., fill_value=...
    ) -> FrameOrSeries: ...
    @final
    def slice_shift(
        self: FrameOrSeries, periods: int = ..., axis=...
    ) -> FrameOrSeries: ...
    @final
    def tshift(
        self: FrameOrSeries, periods: int = ..., freq=..., axis: Axis = ...
    ) -> FrameOrSeries: ...
    def truncate(
        self: FrameOrSeries, before=..., after=..., axis=..., copy: bool_t = ...
    ) -> FrameOrSeries: ...
    @final
    def tz_convert(
        self: FrameOrSeries, tz, axis=..., level=..., copy: bool_t = ...
    ) -> FrameOrSeries: ...
    @final
    def tz_localize(
        self: FrameOrSeries,
        tz,
        axis=...,
        level=...,
        copy: bool_t = ...,
        ambiguous=...,
        nonexistent: str = ...,
    ) -> FrameOrSeries: ...
    @final
    def abs(self: FrameOrSeries) -> FrameOrSeries: ...
    @final
    def describe(
        self: FrameOrSeries,
        percentiles=...,
        include=...,
        exclude=...,
        datetime_is_numeric=...,
    ) -> FrameOrSeries: ...
    @final
    def pct_change(
        self: FrameOrSeries, periods=..., fill_method=..., limit=..., freq=..., **kwargs
    ) -> FrameOrSeries: ...
    def any(self, axis=..., bool_only=..., skipna=..., level=..., **kwargs): ...
    def all(self, axis=..., bool_only=..., skipna=..., level=..., **kwargs): ...
    def cummax(self, axis=..., skipna=..., *args, **kwargs): ...
    def cummin(self, axis=..., skipna=..., *args, **kwargs): ...
    def cumsum(self, axis=..., skipna=..., *args, **kwargs): ...
    def cumprod(self, axis=..., skipna=..., *args, **kwargs): ...
    def sem(
        self, axis=..., skipna=..., level=..., ddof=..., numeric_only=..., **kwargs
    ): ...
    def var(
        self, axis=..., skipna=..., level=..., ddof=..., numeric_only=..., **kwargs
    ): ...
    def std(
        self, axis=..., skipna=..., level=..., ddof=..., numeric_only=..., **kwargs
    ): ...
    def min(self, axis=..., skipna=..., level=..., numeric_only=..., **kwargs): ...
    def max(self, axis=..., skipna=..., level=..., numeric_only=..., **kwargs): ...
    def mean(self, axis=..., skipna=..., level=..., numeric_only=..., **kwargs): ...
    def median(self, axis=..., skipna=..., level=..., numeric_only=..., **kwargs): ...
    def skew(self, axis=..., skipna=..., level=..., numeric_only=..., **kwargs): ...
    def kurt(self, axis=..., skipna=..., level=..., numeric_only=..., **kwargs): ...
    kurtosis = ...
    def sum(
        self, axis=..., skipna=..., level=..., numeric_only=..., min_count=..., **kwargs
    ): ...
    def prod(
        self, axis=..., skipna=..., level=..., numeric_only=..., min_count=..., **kwargs
    ): ...
    product = ...
    def mad(self, axis=..., skipna=..., level=...): ...
    @final
    @doc(Rolling)
    def rolling(
        self,
        window: int | timedelta | BaseOffset | BaseIndexer,
        min_periods: int | None = ...,
        center: bool_t = ...,
        win_type: str | None = ...,
        on: str | None = ...,
        axis: Axis = ...,
        closed: str | None = ...,
        method: str = ...,
    ): ...
    @final
    @doc(Expanding)
    def expanding(
        self,
        min_periods: int = ...,
        center: bool_t | None = ...,
        axis: Axis = ...,
        method: str = ...,
    ) -> Expanding: ...
    @final
    @doc(ExponentialMovingWindow)
    def ewm(
        self,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | TimedeltaConvertibleTypes | None = ...,
        alpha: float | None = ...,
        min_periods: int | None = ...,
        adjust: bool_t = ...,
        ignore_na: bool_t = ...,
        axis: Axis = ...,
        times: str | np.ndarray | FrameOrSeries | None = ...,
    ) -> ExponentialMovingWindow: ...
    def __iadd__(self, other): ...
    def __isub__(self, other): ...
    def __imul__(self, other): ...
    def __itruediv__(self, other): ...
    def __ifloordiv__(self, other): ...
    def __imod__(self, other): ...
    def __ipow__(self, other): ...
    def __iand__(self, other): ...
    def __ior__(self, other): ...
    def __ixor__(self, other): ...
    @final
    @doc(position="first", klass=_shared_doc_kwargs["klass"])
    def first_valid_index(self) -> Hashable | None: ...
    @final
    @doc(first_valid_index, position="last", klass=_shared_doc_kwargs["klass"])
    def last_valid_index(self) -> Hashable | None: ...

_num_doc = ...
_num_ddof_doc = ...
_bool_doc = ...
_all_desc = ...
_all_examples = ...
_all_see_also = ...
_cnum_doc = ...
_cummin_examples = ...
_cumsum_examples = ...
_cumprod_examples = ...
_cummax_examples = ...
_any_see_also = ...
_any_desc = ...
_any_examples = ...
_sum_examples = ...
_max_examples = ...
_min_examples = ...
_stat_func_see_also = ...
_prod_examples = ...
_min_count_stub = ...
