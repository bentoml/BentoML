from typing import Any, Callable, Hashable, Sequence
from pandas._typing import (
    Axis,
    FilePathOrBuffer,
    FrameOrSeriesUnion,
    IndexLabel,
    Scalar,
)
from pandas.core import generic
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.io.formats.style_render import (
    CSSProperties,
    CSSStyles,
    StylerRenderer,
    Subset,
)
from pandas.util._decorators import doc

jinja2 = ...

class Styler(StylerRenderer):
    def __init__(
        self,
        data: FrameOrSeriesUnion,
        precision: int | None = ...,
        table_styles: CSSStyles | None = ...,
        uuid: str | None = ...,
        caption: str | tuple | None = ...,
        table_attributes: str | None = ...,
        cell_ids: bool = ...,
        na_rep: str | None = ...,
        uuid_len: int = ...,
        decimal: str = ...,
        thousands: str | None = ...,
        escape: str | None = ...,
    ) -> None: ...
    def render(
        self,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        **kwargs
    ) -> str: ...
    def set_tooltips(
        self,
        ttips: DataFrame,
        props: CSSProperties | None = ...,
        css_class: str | None = ...,
    ) -> Styler: ...
    @doc(
        NDFrame.to_excel,
        klass="Styler",
        storage_options=generic._shared_docs["storage_options"],
    )
    def to_excel(
        self,
        excel_writer,
        sheet_name: str = ...,
        na_rep: str = ...,
        float_format: str | None = ...,
        columns: Sequence[Hashable] | None = ...,
        header: Sequence[Hashable] | bool = ...,
        index: bool = ...,
        index_label: IndexLabel | None = ...,
        startrow: int = ...,
        startcol: int = ...,
        engine: str | None = ...,
        merge_cells: bool = ...,
        encoding: str | None = ...,
        inf_rep: str = ...,
        verbose: bool = ...,
        freeze_panes: tuple[int, int] | None = ...,
    ) -> None: ...
    def to_latex(
        self,
        buf: FilePathOrBuffer[str] | None = ...,
        *,
        column_format: str | None = ...,
        position: str | None = ...,
        position_float: str | None = ...,
        hrules: bool = ...,
        label: str | None = ...,
        caption: str | tuple | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        multirow_align: str = ...,
        multicol_align: str = ...,
        siunitx: bool = ...,
        encoding: str | None = ...,
        convert_css: bool = ...
    ): ...
    def to_html(
        self,
        buf: FilePathOrBuffer[str] | None = ...,
        *,
        table_uuid: str | None = ...,
        table_attributes: str | None = ...,
        encoding: str | None = ...,
        doctype_html: bool = ...,
        exclude_styles: bool = ...
    ): ...
    def set_td_classes(self, classes: DataFrame) -> Styler: ...
    def __copy__(self) -> Styler: ...
    def __deepcopy__(self, memo) -> Styler: ...
    def clear(self) -> None: ...
    def apply(
        self,
        func: Callable[..., Styler],
        axis: Axis | None = ...,
        subset: Subset | None = ...,
        **kwargs
    ) -> Styler: ...
    def applymap(
        self, func: Callable, subset: Subset | None = ..., **kwargs
    ) -> Styler: ...
    def where(
        self,
        cond: Callable,
        value: str,
        other: str | None = ...,
        subset: Subset | None = ...,
        **kwargs
    ) -> Styler: ...
    def set_precision(self, precision: int) -> StylerRenderer: ...
    def set_table_attributes(self, attributes: str) -> Styler: ...
    def export(self) -> list[tuple[Callable, tuple, dict]]: ...
    def use(self, styles: list[tuple[Callable, tuple, dict]]) -> Styler: ...
    def set_uuid(self, uuid: str) -> Styler: ...
    def set_caption(self, caption: str | tuple) -> Styler: ...
    def set_sticky(
        self,
        axis: Axis = ...,
        pixel_size: int | None = ...,
        levels: list[int] | None = ...,
    ) -> Styler: ...
    def set_table_styles(
        self,
        table_styles: dict[Any, CSSStyles] | CSSStyles,
        axis: int = ...,
        overwrite: bool = ...,
    ) -> Styler: ...
    def set_na_rep(self, na_rep: str) -> StylerRenderer: ...
    def hide_index(self, subset: Subset | None = ...) -> Styler: ...
    def hide_columns(self, subset: Subset | None = ...) -> Styler: ...
    @doc(
        name="background",
        alt="text",
        image_prefix="bg",
        axis="{0 or 'index', 1 or 'columns', None}",
        text_threshold="",
    )
    def background_gradient(
        self,
        cmap=...,
        low: float = ...,
        high: float = ...,
        axis: Axis | None = ...,
        subset: Subset | None = ...,
        text_color_threshold: float = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        gmap: Sequence | None = ...,
    ) -> Styler: ...
    @doc(
        background_gradient,
        name="text",
        alt="background",
        image_prefix="tg",
        axis="{0 or 'index', 1 or 'columns', None}",
        text_threshold="This argument is ignored (only used in `background_gradient`).",
    )
    def text_gradient(
        self,
        cmap=...,
        low: float = ...,
        high: float = ...,
        axis: Axis | None = ...,
        subset: Subset | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        gmap: Sequence | None = ...,
    ) -> Styler: ...
    def set_properties(self, subset: Subset | None = ..., **kwargs) -> Styler: ...
    def bar(
        self,
        subset: Subset | None = ...,
        axis: Axis | None = ...,
        color=...,
        width: float = ...,
        align: str = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
    ) -> Styler: ...
    def highlight_null(
        self,
        null_color: str = ...,
        subset: Subset | None = ...,
        props: str | None = ...,
    ) -> Styler: ...
    def highlight_max(
        self,
        subset: Subset | None = ...,
        color: str = ...,
        axis: Axis | None = ...,
        props: str | None = ...,
    ) -> Styler: ...
    def highlight_min(
        self,
        subset: Subset | None = ...,
        color: str = ...,
        axis: Axis | None = ...,
        props: str | None = ...,
    ) -> Styler: ...
    def highlight_between(
        self,
        subset: Subset | None = ...,
        color: str = ...,
        axis: Axis | None = ...,
        left: Scalar | Sequence | None = ...,
        right: Scalar | Sequence | None = ...,
        inclusive: str = ...,
        props: str | None = ...,
    ) -> Styler: ...
    def highlight_quantile(
        self,
        subset: Subset | None = ...,
        color: str = ...,
        axis: Axis | None = ...,
        q_left: float = ...,
        q_right: float = ...,
        interpolation: str = ...,
        inclusive: str = ...,
        props: str | None = ...,
    ) -> Styler: ...
    @classmethod
    def from_custom_template(
        cls, searchpath, html_table: str | None = ..., html_style: str | None = ...
    ):
        class MyStyler(cls): ...
