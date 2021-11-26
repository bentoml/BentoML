from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from pandas import DataFrame, Index
from pandas._typing import FrameOrSeriesUnion, TypedDict

jinja2 = ...
BaseFormatter = Union[str, Callable]
ExtFormatter = Union[BaseFormatter, Dict[Any, Optional[BaseFormatter]]]
CSSPair = Tuple[str, Union[str, int, float]]
CSSList = List[CSSPair]
CSSProperties = Union[str, CSSList]

class CSSDict(TypedDict):
    selector: str
    props: CSSProperties

...
CSSStyles = List[CSSDict]
Subset = Union[slice, Sequence, Index]

class StylerRenderer:
    loader = ...
    env = ...
    template_html = ...
    template_html_table = ...
    template_html_style = ...
    template_latex = ...
    def __init__(
        self,
        data: FrameOrSeriesUnion,
        uuid: str | None = ...,
        uuid_len: int = ...,
        table_styles: CSSStyles | None = ...,
        table_attributes: str | None = ...,
        caption: str | tuple | None = ...,
        cell_ids: bool = ...,
    ) -> None: ...
    def format(
        self,
        formatter: ExtFormatter | None = ...,
        subset: Subset | None = ...,
        na_rep: str | None = ...,
        precision: int | None = ...,
        decimal: str = ...,
        thousands: str | None = ...,
        escape: str | None = ...,
    ) -> StylerRenderer: ...

def non_reducing_slice(slice_: Subset): ...
def maybe_convert_css_to_tuples(style: CSSProperties) -> CSSList: ...

class Tooltips:
    def __init__(
        self,
        css_props: CSSProperties = ...,
        css_name: str = ...,
        tooltips: DataFrame = ...,
    ) -> None: ...
