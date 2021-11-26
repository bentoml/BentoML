from typing import Any, Callable, Dict, Optional
from ...configuration_utils import PretrainedConfig
from .auto_factory import PathLike

ALL_PRETRAINED_CONFIG_ARCHIVE_MAP: Dict[str, Any] = ...
CONFIG_MAPPING: Dict[str, Any] = ...
MODEL_NAMES_MAPPING: Dict[str, Any] = ...

def replace_list_option_in_docstrings(
    config_to_class: None = ..., use_model_types: bool = ...
) -> Callable[..., Callable[..., Any]]: ...

class AutoConfig:
    def __init__(self) -> None: ...
    @classmethod
    def for_model(
        cls, model_type: str, *args: Any, **kwargs: Any
    ) -> PretrainedConfig: ...
    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: PathLike,
        cache_dir: Optional[PathLike] = ...,
        force_download: Optional[bool] = ...,
        resume_download: Optional[bool] = ...,
        proxies: Optional[str] = ...,
        revision: Optional[str] = ...,
        return_unused_kwargs: Optional[bool] = ...,
        kwargs: Dict[str, Any] = ...,
    ) -> PretrainedConfig: ...
