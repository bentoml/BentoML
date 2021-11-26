import datetime
from textwrap import dedent
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AnyStr,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Optional,
    Sequence,
    overload,
)
import numpy as np
from pandas._libs import lib
from pandas._typing import (
    AggFuncType,
    AnyArrayLike,
    ArrayLike,
    Axes,
    Axis,
    ColspaceArgType,
    CompressionOptions,
    Dtype,
    FilePathOrBuffer,
    FillnaOptions,
    FloatFormatType,
    FormattersType,
    FrameOrSeriesUnion,
    Frequency,
    IndexKeyFunc,
    IndexLabel,
    Level,
    NpDtype,
    PythonFuncType,
    Renamer,
    StorageOptions,
    Suffixes,
    TimedeltaConvertibleTypes,
    TimestampConvertibleTypes,
    ValueKeyFunc,
)
from pandas.core import generic
from pandas.core.arraylike import OpsMixin
from pandas.core.dtypes.missing import isna, notna
from pandas.core.generic import NDFrame, _shared_docs
from pandas.core.groupby.generic import DataFrameGroupBy
from pandas.core.indexes.api import Index
from pandas.core.internals import ArrayManager, BlockManager
from pandas.core.resample import Resampler
from pandas.core.series import Series
from pandas.io.formats import format as fmt
from pandas.io.formats.info import BaseInfo
from pandas.io.formats.style import Styler
from pandas.util._decorators import (
    Appender,
    Substitution,
    deprecate_kwarg,
    deprecate_nonkeyword_arguments,
    doc,
    rewrite_axis_style_signature,
)

class DataFrame(NDFrame, OpsMixin):
    _internal_names_set = ...
    _typ = ...
    _HANDLED_TYPES = ...
    _accessors: set[str] = ...
    _hidden_attrs: frozenset[str] = ...
    _mgr: BlockManager | ArrayManager
    _constructor_sliced: type[Series] = ...
    def __init__(
        self,
        data=...,
        index: Axes | None = ...,
        columns: Axes | None = ...,
        dtype: Dtype | None = ...,
        copy: bool | None = ...,
    ) -> None: ...
    @property
    def axes(self) -> list[Index]: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    def __repr__(self) -> str: ...
    @Substitution(
        header_type="bool or sequence",
        header="""Write out the column names. If a list of strings is given, it is assumed to be aliases for the column names""",
        col_space_type="int, list or dict of int",
        col_space="The minimum width of each column",
    )
    @Substitution(shared_params=fmt.common_docstring, returns=fmt.return_docstring)
    def to_string(
        self,
        buf: FilePathOrBuffer[str] | None = ...,
        columns: Sequence[str] | None = ...,
        col_space: int | None = ...,
        header: bool | Sequence[str] = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: fmt.FormattersType | None = ...,
        float_format: fmt.FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        justify: str | None = ...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: bool = ...,
        decimal: str = ...,
        line_width: int | None = ...,
        max_colwidth: int | None = ...,
        encoding: str | None = ...,
    ) -> str | None: ...
    @property
    def style(self) -> Styler: ...
    @Appender(_shared_docs["items"])
    def items(self) -> Iterable[tuple[Hashable, Series]]: ...
    @Appender(_shared_docs["items"])
    def iteritems(self) -> Iterable[tuple[Hashable, Series]]: ...
    def iterrows(self) -> Iterable[tuple[Hashable, Series]]: ...
    def itertuples(
        self, index: bool = ..., name: str | None = ...
    ) -> Iterable[tuple[Any, ...]]: ...
    def __len__(self) -> int: ...
    @overload
    def dot(self, other: Series) -> Series: ...
    @overload
    def dot(self, other: DataFrame | Index | ArrayLike) -> DataFrame: ...
    def dot(self, other: AnyArrayLike | FrameOrSeriesUnion) -> FrameOrSeriesUnion: ...
    @overload
    def __matmul__(self, other: Series) -> Series: ...
    @overload
    def __matmul__(
        self, other: AnyArrayLike | FrameOrSeriesUnion
    ) -> FrameOrSeriesUnion: ...
    def __matmul__(
        self, other: AnyArrayLike | FrameOrSeriesUnion
    ) -> FrameOrSeriesUnion: ...
    def __rmatmul__(self, other): ...
    @classmethod
    def from_dict(
        cls, data, orient: str = ..., dtype: Dtype | None = ..., columns=...
    ) -> DataFrame: ...
    @overload
    def to_numpy(
        self, dtype: np.dtype[Any] = ..., copy: bool = ..., na_value: Any = ...
    ) -> np.ndarray[Any, np.dtype[Any]]: ...
    @overload
    def to_numpy(
        self, dtype: None = ..., copy: bool = ..., na_value: Any = ...
    ) -> np.ndarray[Any, np.dtype[Any]]: ...
    @overload
    def to_numpy(
        self, dtype: Optional[NpDtype] = ..., copy: bool = ..., na_value: Any = ...
    ) -> np.ndarray[Any, np.dtype[Any]]: ...
    def to_dict(self, orient: str = ..., into=...): ...
    def to_gbq(
        self,
        destination_table: str,
        project_id: str | None = ...,
        chunksize: int | None = ...,
        reauth: bool = ...,
        if_exists: str = ...,
        auth_local_webserver: bool = ...,
        table_schema: list[dict[str, str]] | None = ...,
        location: str | None = ...,
        progress_bar: bool = ...,
        credentials=...,
    ) -> None: ...
    @classmethod
    def from_records(
        cls,
        data,
        index=...,
        exclude=...,
        columns=...,
        coerce_float: bool = ...,
        nrows: int | None = ...,
    ) -> DataFrame: ...
    def to_records(
        self, index=..., column_dtypes=..., index_dtypes=...
    ) -> np.recarray: ...
    @doc(storage_options=generic._shared_docs["storage_options"])
    @deprecate_kwarg(old_arg_name="fname", new_arg_name="path")
    def to_stata(
        self,
        path: FilePathOrBuffer,
        convert_dates: dict[Hashable, str] | None = ...,
        write_index: bool = ...,
        byteorder: str | None = ...,
        time_stamp: datetime.datetime | None = ...,
        data_label: str | None = ...,
        variable_labels: dict[Hashable, str] | None = ...,
        version: int | None = ...,
        convert_strl: Sequence[Hashable] | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    @deprecate_kwarg(old_arg_name="fname", new_arg_name="path")
    def to_feather(self, path: FilePathOrBuffer[AnyStr], **kwargs) -> None: ...
    @doc(
        Series.to_markdown,
        klass=_shared_doc_kwargs["klass"],
        storage_options=_shared_docs["storage_options"],
        examples="""Examples
        --------
        >>> df = pd.DataFrame(
        ...     data={"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]}
        ... )
        >>> print(df.to_markdown())
        |    | animal_1   | animal_2   |
        |---:|:-----------|:-----------|
        |  0 | elk        | dog        |
        |  1 | pig        | quetzal    |
        Output markdown with a tabulate option.
        >>> print(df.to_markdown(tablefmt="grid"))
        +----+------------+------------+
        |    | animal_1   | animal_2   |
        +====+============+============+
        |  0 | elk        | dog        |
        +----+------------+------------+
        |  1 | pig        | quetzal    |
        +----+------------+------------+
        """,
    )
    def to_markdown(
        self,
        buf: IO[str] | str | None = ...,
        mode: str = ...,
        index: bool = ...,
        storage_options: StorageOptions = ...,
        **kwargs
    ) -> str | None: ...
    @doc(storage_options=generic._shared_docs["storage_options"])
    @deprecate_kwarg(old_arg_name="fname", new_arg_name="path")
    def to_parquet(
        self,
        path: FilePathOrBuffer | None = ...,
        engine: str = ...,
        compression: str | None = ...,
        index: bool | None = ...,
        partition_cols: list[str] | None = ...,
        storage_options: StorageOptions = ...,
        **kwargs
    ) -> bytes | None: ...
    @Substitution(
        header_type="bool",
        header="Whether to print column labels, default True",
        col_space_type="str or int, list or dict of int or str",
        col_space="""The minimum width of each column in CSS length units.  An int is assumed to be px units.\n\n            .. versionadded:: 0.25.0\n                Ability to use str""",
    )
    @Substitution(shared_params=fmt.common_docstring, returns=fmt.return_docstring)
    def to_html(
        self,
        buf: FilePathOrBuffer[str] | None = ...,
        columns: Sequence[str] | None = ...,
        col_space: ColspaceArgType | None = ...,
        header: bool | Sequence[str] = ...,
        index: bool = ...,
        na_rep: str = ...,
        formatters: FormattersType | None = ...,
        float_format: FloatFormatType | None = ...,
        sparsify: bool | None = ...,
        index_names: bool = ...,
        justify: str | None = ...,
        max_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: bool | str = ...,
        decimal: str = ...,
        bold_rows: bool = ...,
        classes: str | list | tuple | None = ...,
        escape: bool = ...,
        notebook: bool = ...,
        border: int | None = ...,
        table_id: str | None = ...,
        render_links: bool = ...,
        encoding: str | None = ...,
    ): ...
    @doc(storage_options=generic._shared_docs["storage_options"])
    def to_xml(
        self,
        path_or_buffer: FilePathOrBuffer | None = ...,
        index: bool = ...,
        root_name: str | None = ...,
        row_name: str | None = ...,
        na_rep: str | None = ...,
        attr_cols: str | list[str] | None = ...,
        elem_cols: str | list[str] | None = ...,
        namespaces: dict[str | None, str] | None = ...,
        prefix: str | None = ...,
        encoding: str = ...,
        xml_declaration: bool | None = ...,
        pretty_print: bool | None = ...,
        parser: str | None = ...,
        stylesheet: FilePathOrBuffer | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
    ) -> str | None: ...
    @Substitution(
        klass="DataFrame",
        type_sub=" and columns",
        max_cols_sub=dedent(
            """            max_cols : int, optional
                When to switch from the verbose to the truncated output. If the
                DataFrame has more than `max_cols` columns, the truncated output
                is used. By default, the setting in
                ``pandas.options.display.max_info_columns`` is used."""
        ),
        show_counts_sub=dedent(
            """            show_counts : bool, optional
                Whether to show the non-null counts. By default, this is shown
                only if the DataFrame is smaller than
                ``pandas.options.display.max_info_rows`` and
                ``pandas.options.display.max_info_columns``. A value of True always
                shows the counts, and False never shows the counts.
            null_counts : bool, optional
                .. deprecated:: 1.2.0
                    Use show_counts instead."""
        ),
        examples_sub=dedent(
            """            >>> int_values = [1, 2, 3, 4, 5]
            >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
            >>> float_values = [0.0, 0.25, 0.5, 0.75, 1.0]
            >>> df = pd.DataFrame({"int_col": int_values, "text_col": text_values,
            ...                   "float_col": float_values})
            >>> df
                int_col text_col  float_col
            0        1    alpha       0.00
            1        2     beta       0.25
            2        3    gamma       0.50
            3        4    delta       0.75
            4        5  epsilon       1.00
            Prints information of all columns:
            >>> df.info(verbose=True)
            <class 'pandas.core.frame.DataFrame'>
            RangeIndex: 5 entries, 0 to 4
            Data columns (total 3 columns):
             #   Column     Non-Null Count  Dtype
            ---  ------     --------------  -----
             0   int_col    5 non-null      int64
             1   text_col   5 non-null      object
             2   float_col  5 non-null      float64
            dtypes: float64(1), int64(1), object(1)
            memory usage: 248.0+ bytes
            Prints a summary of columns count and its dtypes but not per column
            information:
            >>> df.info(verbose=False)
            <class 'pandas.core.frame.DataFrame'>
            RangeIndex: 5 entries, 0 to 4
            Columns: 3 entries, int_col to float_col
            dtypes: float64(1), int64(1), object(1)
            memory usage: 248.0+ bytes
            Pipe output of DataFrame.info to buffer instead of sys.stdout, get
            buffer content and writes to a text file:
            >>> import io
            >>> buffer = io.StringIO()
            >>> df.info(buf=buffer)
            >>> s = buffer.getvalue()
            >>> with open("df_info.txt", "w",
            ...           encoding="utf-8") as f:  # doctest: +SKIP
            ...     f.write(s)
            260
            The `memory_usage` parameter allows deep introspection mode, specially
            useful for big DataFrames and fine-tune memory optimization:
            >>> random_strings_array = np.random.choice(['a', 'b', 'c'], 10 ** 6)
            >>> df = pd.DataFrame({
            ...     'column_1': np.random.choice(['a', 'b', 'c'], 10 ** 6),
            ...     'column_2': np.random.choice(['a', 'b', 'c'], 10 ** 6),
            ...     'column_3': np.random.choice(['a', 'b', 'c'], 10 ** 6)
            ... })
            >>> df.info()
            <class 'pandas.core.frame.DataFrame'>
            RangeIndex: 1000000 entries, 0 to 999999
            Data columns (total 3 columns):
             #   Column    Non-Null Count    Dtype
            ---  ------    --------------    -----
             0   column_1  1000000 non-null  object
             1   column_2  1000000 non-null  object
             2   column_3  1000000 non-null  object
            dtypes: object(3)
            memory usage: 22.9+ MB
            >>> df.info(memory_usage='deep')
            <class 'pandas.core.frame.DataFrame'>
            RangeIndex: 1000000 entries, 0 to 999999
            Data columns (total 3 columns):
             #   Column    Non-Null Count    Dtype
            ---  ------    --------------    -----
             0   column_1  1000000 non-null  object
             1   column_2  1000000 non-null  object
             2   column_3  1000000 non-null  object
            dtypes: object(3)
            memory usage: 165.9 MB"""
        ),
        see_also_sub=dedent(
            """            DataFrame.describe: Generate descriptive statistics of DataFrame
                columns.
            DataFrame.memory_usage: Memory usage of DataFrame columns."""
        ),
        version_added_sub="",
    )
    @doc(BaseInfo.render)
    def info(
        self,
        verbose: bool | None = ...,
        buf: IO[str] | None = ...,
        max_cols: int | None = ...,
        memory_usage: bool | str | None = ...,
        show_counts: bool | None = ...,
        null_counts: bool | None = ...,
    ) -> None: ...
    def memory_usage(self, index: bool = ..., deep: bool = ...) -> Series: ...
    def transpose(self, *args, copy: bool = ...) -> DataFrame: ...
    @property
    def T(self) -> DataFrame: ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value): ...
    def query(self, expr: str, inplace: bool = ..., **kwargs): ...
    def eval(self, expr: str, inplace: bool = ..., **kwargs): ...
    def select_dtypes(self, include=..., exclude=...) -> DataFrame: ...
    def insert(self, loc, column, value, allow_duplicates: bool = ...) -> None: ...
    def assign(self, **kwargs) -> DataFrame: ...
    def lookup(
        self, row_labels: Sequence[IndexLabel], col_labels: Sequence[IndexLabel]
    ) -> np.ndarray: ...
    @doc(NDFrame.align, **_shared_doc_kwargs)
    def align(
        self,
        other,
        join: str = ...,
        axis: Axis | None = ...,
        level: Level | None = ...,
        copy: bool = ...,
        fill_value=...,
        method: str | None = ...,
        limit=...,
        fill_axis: Axis = ...,
        broadcast_axis: Axis | None = ...,
    ) -> DataFrame: ...
    @overload
    def set_axis(
        self, labels, axis: Axis = ..., inplace: Literal[False] = ...
    ) -> DataFrame: ...
    @overload
    def set_axis(self, labels, axis: Axis, inplace: Literal[True]) -> None: ...
    @overload
    def set_axis(self, labels, *, inplace: Literal[True]) -> None: ...
    @overload
    def set_axis(
        self, labels, axis: Axis = ..., inplace: bool = ...
    ) -> DataFrame | None: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "labels"])
    @Appender(
        """
        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        Change the row labels.
        >>> df.set_axis(['a', 'b', 'c'], axis='index')
           A  B
        a  1  4
        b  2  5
        c  3  6
        Change the column labels.
        >>> df.set_axis(['I', 'II'], axis='columns')
           I  II
        0  1   4
        1  2   5
        2  3   6
        Now, update the labels inplace.
        >>> df.set_axis(['i', 'ii'], axis='columns', inplace=True)
        >>> df
           i  ii
        0  1   4
        1  2   5
        2  3   6
        """
    )
    @Substitution(
        **_shared_doc_kwargs,
        extended_summary_sub=" column or",
        axis_description_sub=", and 1 identifies the columns",
        see_also_sub=" or columns"
    )
    @Appender(NDFrame.set_axis.__doc__)
    def set_axis(self, labels, axis: Axis = ..., inplace: bool = ...): ...
    @Substitution(**_shared_doc_kwargs)
    @Appender(NDFrame.reindex.__doc__)
    @rewrite_axis_style_signature(
        "labels",
        [
            ("method", None),
            ("copy", True),
            ("level", None),
            ("fill_value", np.nan),
            ("limit", None),
            ("tolerance", None),
        ],
    )
    def reindex(self, *args, **kwargs) -> DataFrame: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "labels"])
    def drop(
        self,
        labels=...,
        axis: Axis = ...,
        index=...,
        columns=...,
        level: Level | None = ...,
        inplace: bool = ...,
        errors: str = ...,
    ): ...
    @rewrite_axis_style_signature(
        "mapper",
        [("copy", True), ("inplace", False), ("level", None), ("errors", "ignore")],
    )
    def rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: bool = ...,
        level: Level | None = ...,
        errors: str = ...
    ) -> DataFrame | None: ...
    @overload
    def fillna(
        self,
        value=...,
        method: FillnaOptions | None = ...,
        axis: Axis | None = ...,
        inplace: Literal[False] = ...,
        limit=...,
        downcast=...,
    ) -> DataFrame: ...
    @overload
    def fillna(
        self,
        value,
        method: FillnaOptions | None,
        axis: Axis | None,
        inplace: Literal[True],
        limit=...,
        downcast=...,
    ) -> None: ...
    @overload
    def fillna(self, *, inplace: Literal[True], limit=..., downcast=...) -> None: ...
    @overload
    def fillna(
        self, value, *, inplace: Literal[True], limit=..., downcast=...
    ) -> None: ...
    @overload
    def fillna(
        self,
        *,
        method: FillnaOptions | None,
        inplace: Literal[True],
        limit=...,
        downcast=...
    ) -> None: ...
    @overload
    def fillna(
        self, *, axis: Axis | None, inplace: Literal[True], limit=..., downcast=...
    ) -> None: ...
    @overload
    def fillna(
        self,
        *,
        method: FillnaOptions | None,
        axis: Axis | None,
        inplace: Literal[True],
        limit=...,
        downcast=...
    ) -> None: ...
    @overload
    def fillna(
        self,
        value,
        *,
        axis: Axis | None,
        inplace: Literal[True],
        limit=...,
        downcast=...
    ) -> None: ...
    @overload
    def fillna(
        self,
        value,
        method: FillnaOptions | None,
        *,
        inplace: Literal[True],
        limit=...,
        downcast=...
    ) -> None: ...
    @overload
    def fillna(
        self,
        value=...,
        method: FillnaOptions | None = ...,
        axis: Axis | None = ...,
        inplace: bool = ...,
        limit=...,
        downcast=...,
    ) -> DataFrame | None: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "value"])
    @doc(NDFrame.fillna, **_shared_doc_kwargs)
    def fillna(
        self,
        value: object | ArrayLike | None = ...,
        method: FillnaOptions | None = ...,
        axis: Axis | None = ...,
        inplace: bool = ...,
        limit=...,
        downcast=...,
    ) -> DataFrame | None: ...
    def pop(self, item: Hashable) -> Series: ...
    @doc(NDFrame.replace, **_shared_doc_kwargs)
    def replace(
        self,
        to_replace=...,
        value=...,
        inplace: bool = ...,
        limit=...,
        regex: bool = ...,
        method: str = ...,
    ): ...
    @doc(NDFrame.shift, klass=_shared_doc_kwargs["klass"])
    def shift(
        self,
        periods=...,
        freq: Frequency | None = ...,
        axis: Axis = ...,
        fill_value=...,
    ) -> DataFrame: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "keys"])
    def set_index(
        self,
        keys,
        drop: bool = ...,
        append: bool = ...,
        inplace: bool = ...,
        verify_integrity: bool = ...,
    ): ...
    @overload
    def reset_index(
        self,
        level: Hashable | Sequence[Hashable] | None = ...,
        drop: bool = ...,
        inplace: Literal[False] = ...,
        col_level: Hashable = ...,
        col_fill: Hashable = ...,
    ) -> DataFrame: ...
    @overload
    def reset_index(
        self,
        level: Hashable | Sequence[Hashable] | None,
        drop: bool,
        inplace: Literal[True],
        col_level: Hashable = ...,
        col_fill: Hashable = ...,
    ) -> None: ...
    @overload
    def reset_index(
        self,
        *,
        drop: bool,
        inplace: Literal[True],
        col_level: Hashable = ...,
        col_fill: Hashable = ...
    ) -> None: ...
    @overload
    def reset_index(
        self,
        level: Hashable | Sequence[Hashable] | None,
        *,
        inplace: Literal[True],
        col_level: Hashable = ...,
        col_fill: Hashable = ...
    ) -> None: ...
    @overload
    def reset_index(
        self,
        *,
        inplace: Literal[True],
        col_level: Hashable = ...,
        col_fill: Hashable = ...
    ) -> None: ...
    @overload
    def reset_index(
        self,
        level: Hashable | Sequence[Hashable] | None = ...,
        drop: bool = ...,
        inplace: bool = ...,
        col_level: Hashable = ...,
        col_fill: Hashable = ...,
    ) -> DataFrame | None: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "level"])
    def reset_index(
        self,
        level: Hashable | Sequence[Hashable] | None = ...,
        drop: bool = ...,
        inplace: bool = ...,
        col_level: Hashable = ...,
        col_fill: Hashable = ...,
    ) -> DataFrame | None: ...
    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])
    def isna(self) -> DataFrame: ...
    @doc(NDFrame.isna, klass=_shared_doc_kwargs["klass"])
    def isnull(self) -> DataFrame: ...
    @doc(NDFrame.notna, klass=_shared_doc_kwargs["klass"])
    def notna(self) -> DataFrame: ...
    @doc(NDFrame.notna, klass=_shared_doc_kwargs["klass"])
    def notnull(self) -> DataFrame: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def dropna(
        self,
        axis: Axis = ...,
        how: str = ...,
        thresh=...,
        subset=...,
        inplace: bool = ...,
    ): ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "subset"])
    def drop_duplicates(
        self,
        subset: Hashable | Sequence[Hashable] | None = ...,
        keep: Literal["first"] | Literal["last"] | Literal[False] = ...,
        inplace: bool = ...,
        ignore_index: bool = ...,
    ) -> DataFrame | None: ...
    def duplicated(
        self,
        subset: Hashable | Sequence[Hashable] | None = ...,
        keep: Literal["first"] | Literal["last"] | Literal[False] = ...,
    ) -> Series: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "by"])
    @Substitution(**_shared_doc_kwargs)
    @Appender(NDFrame.sort_values.__doc__)
    def sort_values(
        self,
        by,
        axis: Axis = ...,
        ascending=...,
        inplace: bool = ...,
        kind: str = ...,
        na_position: str = ...,
        ignore_index: bool = ...,
        key: ValueKeyFunc = ...,
    ): ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def sort_index(
        self,
        axis: Axis = ...,
        level: Level | None = ...,
        ascending: bool | int | Sequence[bool | int] = ...,
        inplace: bool = ...,
        kind: str = ...,
        na_position: str = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key: IndexKeyFunc = ...,
    ): ...
    def value_counts(
        self,
        subset: Sequence[Hashable] | None = ...,
        normalize: bool = ...,
        sort: bool = ...,
        ascending: bool = ...,
        dropna: bool = ...,
    ): ...
    def nlargest(self, n, columns, keep: str = ...) -> DataFrame: ...
    def nsmallest(self, n, columns, keep: str = ...) -> DataFrame: ...
    @doc(
        Series.swaplevel,
        klass=_shared_doc_kwargs["klass"],
        extra_params=dedent(
            """axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to swap levels on. 0 or 'index' for row-wise, 1 or
            'columns' for column-wise."""
        ),
        examples=dedent(
            """Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"Grade": ["A", "B", "A", "C"]},
        ...     index=[
        ...         ["Final exam", "Final exam", "Coursework", "Coursework"],
        ...         ["History", "Geography", "History", "Geography"],
        ...         ["January", "February", "March", "April"],
        ...     ],
        ... )
        >>> df
                                            Grade
        Final exam  History     January      A
                    Geography   February     B
        Coursework  History     March        A
                    Geography   April        C
        In the following example, we will swap the levels of the indices.
        Here, we will swap the levels column-wise, but levels can be swapped row-wise
        in a similar manner. Note that column-wise is the default behaviour.
        By not supplying any arguments for i and j, we swap the last and second to
        last indices.
        >>> df.swaplevel()
                                            Grade
        Final exam  January     History         A
                    February    Geography       B
        Coursework  March       History         A
                    April       Geography       C
        By supplying one argument, we can choose which index to swap the last
        index with. We can for example swap the first index with the last one as
        follows.
        >>> df.swaplevel(0)
                                            Grade
        January     History     Final exam      A
        February    Geography   Final exam      B
        March       History     Coursework      A
        April       Geography   Coursework      C
        We can also define explicitly which indices we want to swap by supplying values
        for both i and j. Here, we for example swap the first and second indices.
        >>> df.swaplevel(0, 1)
                                            Grade
        History     Final exam  January         A
        Geography   Final exam  February        B
        History     Coursework  March           A
        Geography   Coursework  April           C"""
        ),
    )
    def swaplevel(
        self, i: Axis = ..., j: Axis = ..., axis: Axis = ...
    ) -> DataFrame: ...
    def reorder_levels(self, order: Sequence[Axis], axis: Axis = ...) -> DataFrame: ...
    _logical_method = ...
    def __divmod__(self, other) -> tuple[DataFrame, DataFrame]: ...
    def __rdivmod__(self, other) -> tuple[DataFrame, DataFrame]: ...
    def compare(
        self,
        other: DataFrame,
        align_axis: Axis = ...,
        keep_shape: bool = ...,
        keep_equal: bool = ...,
    ) -> DataFrame: ...
    def combine(
        self, other: DataFrame, func, fill_value=..., overwrite: bool = ...
    ) -> DataFrame: ...
    def combine_first(self, other: DataFrame) -> DataFrame: ...
    def update(
        self,
        other,
        join: str = ...,
        overwrite: bool = ...,
        filter_func=...,
        errors: str = ...,
    ) -> None: ...
    @Appender(_shared_docs["groupby"] % _shared_doc_kwargs)
    def groupby(
        self,
        by=...,
        axis: Axis = ...,
        level: Level | None = ...,
        as_index: bool = ...,
        sort: bool = ...,
        group_keys: bool = ...,
        squeeze: bool | lib.NoDefault = ...,
        observed: bool = ...,
        dropna: bool = ...,
    ) -> DataFrameGroupBy: ...
    @Substitution("")
    @Appender(_shared_docs["pivot"])
    def pivot(self, index=..., columns=..., values=...) -> DataFrame: ...
    @Substitution("")
    @Appender(_shared_docs["pivot_table"])
    def pivot_table(
        self,
        values=...,
        index=...,
        columns=...,
        aggfunc=...,
        fill_value=...,
        margins=...,
        dropna=...,
        margins_name=...,
        observed=...,
        sort=...,
    ) -> DataFrame: ...
    def stack(self, level: Level = ..., dropna: bool = ...): ...
    def explode(self, column: IndexLabel, ignore_index: bool = ...) -> DataFrame: ...
    def unstack(self, level: Level = ..., fill_value=...): ...
    @Appender(_shared_docs["melt"] % {"caller": "df.melt(", "other": "melt"})
    def melt(
        self,
        id_vars=...,
        value_vars=...,
        var_name=...,
        value_name=...,
        col_level: Level | None = ...,
        ignore_index: bool = ...,
    ) -> DataFrame: ...
    @doc(
        Series.diff,
        klass="Dataframe",
        extra_params="""axis : {0 or 'index', 1 or 'columns'}, default 0\n    Take difference over rows (0) or columns (1).\n""",
        other_klass="Series",
        examples=dedent(
            """
        Difference with previous row
        >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
        ...                    'b': [1, 1, 2, 3, 5, 8],
        ...                    'c': [1, 4, 9, 16, 25, 36]})
        >>> df
           a  b   c
        0  1  1   1
        1  2  1   4
        2  3  2   9
        3  4  3  16
        4  5  5  25
        5  6  8  36
        >>> df.diff()
             a    b     c
        0  NaN  NaN   NaN
        1  1.0  0.0   3.0
        2  1.0  1.0   5.0
        3  1.0  1.0   7.0
        4  1.0  2.0   9.0
        5  1.0  3.0  11.0
        Difference with previous column
        >>> df.diff(axis=1)
            a  b   c
        0 NaN  0   0
        1 NaN -1   3
        2 NaN -1   7
        3 NaN -1  13
        4 NaN  0  20
        5 NaN  2  28
        Difference with 3rd previous row
        >>> df.diff(periods=3)
             a    b     c
        0  NaN  NaN   NaN
        1  NaN  NaN   NaN
        2  NaN  NaN   NaN
        3  3.0  2.0  15.0
        4  3.0  4.0  21.0
        5  3.0  6.0  27.0
        Difference with following row
        >>> df.diff(periods=-1)
             a    b     c
        0 -1.0  0.0  -3.0
        1 -1.0 -1.0  -5.0
        2 -1.0 -1.0  -7.0
        3 -1.0 -2.0  -9.0
        4 -1.0 -3.0 -11.0
        5  NaN  NaN   NaN
        Overflow in input dtype
        >>> df = pd.DataFrame({'a': [1, 0]}, dtype=np.uint8)
        >>> df.diff()
               a
        0    NaN
        1  255.0"""
        ),
    )
    def diff(self, periods: int = ..., axis: Axis = ...) -> DataFrame: ...
    _agg_summary_and_see_also_doc = ...
    _agg_examples_doc = ...
    @doc(
        _shared_docs["aggregate"],
        klass=_shared_doc_kwargs["klass"],
        axis=_shared_doc_kwargs["axis"],
        see_also=_agg_summary_and_see_also_doc,
        examples=_agg_examples_doc,
    )
    def aggregate(self, func=..., axis: Axis = ..., *args, **kwargs): ...
    agg = ...
    @doc(
        _shared_docs["transform"],
        klass=_shared_doc_kwargs["klass"],
        axis=_shared_doc_kwargs["axis"],
    )
    def transform(
        self, func: AggFuncType, axis: Axis = ..., *args, **kwargs
    ) -> DataFrame: ...
    def apply(
        self,
        func: AggFuncType,
        axis: Axis = ...,
        raw: bool = ...,
        result_type=...,
        args=...,
        **kwargs
    ): ...
    def applymap(
        self, func: PythonFuncType, na_action: str | None = ..., **kwargs
    ) -> DataFrame: ...
    def append(
        self,
        other,
        ignore_index: bool = ...,
        verify_integrity: bool = ...,
        sort: bool = ...,
    ) -> DataFrame: ...
    def join(
        self,
        other: FrameOrSeriesUnion,
        on: IndexLabel | None = ...,
        how: str = ...,
        lsuffix: str = ...,
        rsuffix: str = ...,
        sort: bool = ...,
    ) -> DataFrame: ...
    @Substitution("")
    @Appender(_merge_doc, indents=2)
    def merge(
        self,
        right: FrameOrSeriesUnion,
        how: str = ...,
        on: IndexLabel | None = ...,
        left_on: IndexLabel | None = ...,
        right_on: IndexLabel | None = ...,
        left_index: bool = ...,
        right_index: bool = ...,
        sort: bool = ...,
        suffixes: Suffixes = ...,
        copy: bool = ...,
        indicator: bool = ...,
        validate: str | None = ...,
    ) -> DataFrame: ...
    def round(
        self, decimals: int | dict[IndexLabel, int] | Series = ..., *args, **kwargs
    ) -> DataFrame: ...
    def corr(
        self,
        method: str | Callable[[np.ndarray, np.ndarray], float] = ...,
        min_periods: int = ...,
    ) -> DataFrame: ...
    def cov(
        self, min_periods: int | None = ..., ddof: int | None = ...
    ) -> DataFrame: ...
    def corrwith(self, other, axis: Axis = ..., drop=..., method=...) -> Series: ...
    def count(
        self, axis: Axis = ..., level: Level | None = ..., numeric_only: bool = ...
    ): ...
    def nunique(self, axis: Axis = ..., dropna: bool = ...) -> Series: ...
    def idxmin(self, axis: Axis = ..., skipna: bool = ...) -> Series: ...
    def idxmax(self, axis: Axis = ..., skipna: bool = ...) -> Series: ...
    def mode(
        self, axis: Axis = ..., numeric_only: bool = ..., dropna: bool = ...
    ) -> DataFrame: ...
    def quantile(
        self,
        q=...,
        axis: Axis = ...,
        numeric_only: bool = ...,
        interpolation: str = ...,
    ): ...
    @doc(NDFrame.asfreq, **_shared_doc_kwargs)
    def asfreq(
        self,
        freq: Frequency,
        method=...,
        how: str | None = ...,
        normalize: bool = ...,
        fill_value=...,
    ) -> DataFrame: ...
    @doc(NDFrame.resample, **_shared_doc_kwargs)
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
    def to_timestamp(
        self,
        freq: Frequency | None = ...,
        how: str = ...,
        axis: Axis = ...,
        copy: bool = ...,
    ) -> DataFrame: ...
    def to_period(
        self, freq: Frequency | None = ..., axis: Axis = ..., copy: bool = ...
    ) -> DataFrame: ...
    def isin(self, values) -> DataFrame: ...
    _AXIS_ORDERS = ...
    _AXIS_TO_AXIS_NUMBER: dict[Axis, int] = ...
    _AXIS_REVERSED = ...
    _AXIS_LEN = ...
    _info_axis_number = ...
    _info_axis_name = ...
    index: Index = ...
    columns: Index = ...
    plot = ...
    hist = ...
    boxplot = ...
    sparse = ...
    @property
    def values(self) -> np.ndarray: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def ffill(
        self: DataFrame,
        axis: None | Axis = ...,
        inplace: bool = ...,
        limit: None | int = ...,
        downcast=...,
    ) -> DataFrame | None: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self"])
    def bfill(
        self: DataFrame,
        axis: None | Axis = ...,
        inplace: bool = ...,
        limit: None | int = ...,
        downcast=...,
    ) -> DataFrame | None: ...
    @deprecate_nonkeyword_arguments(
        version=None, allowed_args=["self", "lower", "upper"]
    )
    def clip(
        self: DataFrame,
        lower=...,
        upper=...,
        axis: Axis | None = ...,
        inplace: bool = ...,
        *args,
        **kwargs
    ) -> DataFrame | None: ...
    @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "method"])
    def interpolate(
        self: DataFrame,
        method: str = ...,
        axis: Axis = ...,
        limit: int | None = ...,
        inplace: bool = ...,
        limit_direction: str | None = ...,
        limit_area: str | None = ...,
        downcast: str | None = ...,
        **kwargs
    ) -> DataFrame | None: ...
    @deprecate_nonkeyword_arguments(
        version=None, allowed_args=["self", "cond", "other"]
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
    @deprecate_nonkeyword_arguments(
        version=None, allowed_args=["self", "cond", "other"]
    )
    def mask(
        self,
        cond,
        other=np.nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=lib.no_default,
    ) -> Any: ...
