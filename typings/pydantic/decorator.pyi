"""
This type stub file was generated by pyright.
"""

from typing import (TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type,
                    Union, overload)

from .main import BaseModel

__all__ = ("validate_arguments",)
if TYPE_CHECKING:
    AnyCallableT = ...
    ConfigType = Union[None, Type[Any], Dict[str, Any]]

@overload
def validate_arguments(
    func: None = ..., *, config: ConfigType = ...
) -> Callable[[AnyCallableT], AnyCallableT]: ...
@overload
def validate_arguments(func: AnyCallableT) -> AnyCallableT: ...
def validate_arguments(
    func: Optional[AnyCallableT] = ..., *, config: ConfigType = ...
) -> Any:
    """
    Decorator to validate the arguments passed to a function.
    """
    ...

ALT_V_ARGS = ...
ALT_V_KWARGS = ...
V_POSITIONAL_ONLY_NAME = ...
V_DUPLICATE_KWARGS = ...

class ValidatedFunction:
    def __init__(self, function: AnyCallableT, config: ConfigType) -> None: ...
    def init_model_instance(self, *args: Any, **kwargs: Any) -> BaseModel: ...
    def call(self, *args: Any, **kwargs: Any) -> Any: ...
    def build_values(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]: ...
    def execute(self, m: BaseModel) -> Any: ...
    def create_model(
        self,
        fields: Dict[str, Any],
        takes_args: bool,
        takes_kwargs: bool,
        config: ConfigType,
    ) -> None:
        class CustomConfig: ...
        class DecoratorBaseModel(BaseModel):
            class Config(CustomConfig): ...
