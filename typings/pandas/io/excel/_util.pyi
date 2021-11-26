from typing import Any, MutableMapping

_writers: MutableMapping[str, str] = ...

def register_writer(klass): ...
def get_default_engine(ext, mode=...): ...
def get_writer(engine_name): ...
def maybe_convert_usecols(usecols): ...
def validate_freeze_panes(freeze_panes): ...
def fill_mi_header(row, control_row): ...
def pop_header_name(row, index_col): ...
def combine_kwargs(engine_kwargs: dict[str, Any] | None, kwargs: dict) -> dict: ...
