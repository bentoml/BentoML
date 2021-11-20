"""
This type stub file was generated by pyright.
"""

from abc import ABCMeta
from enum import Enum
from inspect import Signature
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    no_type_check,
    overload,
)

import typing_extensions

from .error_wrappers import ValidationError
from .fields import ModelField
from .parse import Protocol
from .types import ModelOrDc, StrBytes
from .typing import (
    AbstractSetIntStr,
    AnyCallable,
    CallableGenerator,
    DictAny,
    DictStrAny,
    MappingIntStrAny,
    ReprArgs,
    SetStr,
    TupleGenerator,
)
from .utils import GetterDict, Representation

if TYPE_CHECKING:
    ConfigType = Type["BaseConfig"]
    class SchemaExtraCallable(typing_extensions.Protocol):
        @overload
        def __call__(self, schema: Dict[str, Any]) -> None: ...
        @overload
        def __call__(
            self, schema: Dict[str, Any], model_class: Type["BaseModel"]
        ) -> None: ...

else: ...
__all__ = (
    "BaseConfig",
    "BaseModel",
    "Extra",
    "compiled",
    "create_model",
    "validate_model",
)

class Extra(str, Enum):
    allow = ...
    ignore = ...
    forbid = ...

class BaseConfig:
    title = ...
    anystr_lower = ...
    anystr_strip_whitespace = ...
    min_anystr_length = ...
    max_anystr_length = ...
    validate_all = ...
    extra = ...
    allow_mutation = ...
    frozen = ...
    allow_population_by_field_name = ...
    use_enum_values = ...
    fields: Dict[str, Union[str, Dict[str, str]]] = ...
    validate_assignment = ...
    error_msg_templates: Dict[str, str] = ...
    arbitrary_types_allowed = ...
    orm_mode: bool = ...
    getter_dict: Type[GetterDict] = ...
    alias_generator: Optional[Callable[[str], str]] = ...
    keep_untouched: Tuple[type, ...] = ...
    schema_extra: Union[Dict[str, Any], SchemaExtraCallable] = ...
    json_loads: Callable[[str], Any] = ...
    json_dumps: Callable[..., str] = ...
    json_encoders: Dict[Type[Any], AnyCallable] = ...
    underscore_attrs_are_private: bool = ...
    copy_on_model_validation: bool = ...
    @classmethod
    def get_field_info(cls, name: str) -> Dict[str, Any]:
        """
        Get properties of FieldInfo from the `fields` property of the config class.
        """
        ...
    @classmethod
    def prepare_field(cls, field: ModelField) -> None:
        """
        Optional hook to check or modify fields during model creation.
        """
        ...

def inherit_config(
    self_config: ConfigType, parent_config: ConfigType, **namespace: Any
) -> ConfigType: ...

EXTRA_LINK = ...

def prepare_config(config: Type[BaseConfig], cls_name: str) -> None: ...
def validate_custom_root_type(fields: Dict[str, ModelField]) -> None: ...
def generate_hash_function(frozen: bool) -> Optional[Callable[[Any], int]]: ...

ANNOTATED_FIELD_UNTOUCHED_TYPES: Tuple[Any, ...] = ...
UNTOUCHED_TYPES: Tuple[Any, ...] = ...
_is_base_model_class_defined = ...

class ModelMetaclass(ABCMeta):
    @no_type_check
    def __new__(mcs, name, bases, namespace, **kwargs): ...

object_setattr = ...

class BaseModel(Representation, metaclass=ModelMetaclass):
    if TYPE_CHECKING:
        __fields__: Dict[str, ModelField] = ...
        __validators__: Dict[str, AnyCallable] = ...
        __pre_root_validators__: List[AnyCallable]
        __post_root_validators__: List[Tuple[bool, AnyCallable]]
        __config__: Type[BaseConfig] = ...
        __root__: Any = ...
        __json_encoder__: Callable[[Any], Any] = ...
        __schema_cache__: DictAny = ...
        __custom_root_type__: bool = ...
        __signature__: Signature
        __private_attributes__: Dict[str, Any]
        __class_vars__: SetStr
        __fields_set__: SetStr = ...
    Config = BaseConfig
    __slots__ = ...
    __doc__ = ...
    def __init__(__pydantic_self__, **data: Any) -> None:
        """
        Create a new model by parsing and validating input data from keyword arguments.

        Raises ValidationError if the input data cannot be parsed to form a valid model.
        """
        ...
    @no_type_check
    def __setattr__(self, name, value): ...
    def __getstate__(self) -> DictAny: ...
    def __setstate__(self, state: DictAny) -> None: ...
    def dict(
        self,
        *,
        include: Union[AbstractSetIntStr, MappingIntStrAny] = ...,
        exclude: Union[AbstractSetIntStr, MappingIntStrAny] = ...,
        by_alias: bool = ...,
        skip_defaults: bool = ...,
        exclude_unset: bool = ...,
        exclude_defaults: bool = ...,
        exclude_none: bool = ...
    ) -> DictStrAny:
        """
        Generate a dictionary representation of the model, optionally specifying which fields to include or exclude.

        """
        ...
    def json(
        self,
        *,
        include: Union[AbstractSetIntStr, MappingIntStrAny] = ...,
        exclude: Union[AbstractSetIntStr, MappingIntStrAny] = ...,
        by_alias: bool = ...,
        skip_defaults: bool = ...,
        exclude_unset: bool = ...,
        exclude_defaults: bool = ...,
        exclude_none: bool = ...,
        encoder: Optional[Callable[[Any], Any]] = ...,
        **dumps_kwargs: Any
    ) -> str:
        """
        Generate a JSON representation of the model, `include` and `exclude` arguments as per `dict()`.

        `encoder` is an optional function to supply as `default` to json.dumps(), other arguments as per `json.dumps()`.
        """
        ...
    @classmethod
    def parse_obj(cls: Type[BaseModel], obj: Any) -> Type[BaseModel]: ...
    @classmethod
    def parse_raw(
        cls: Type["BaseModel"],
        b: StrBytes,
        *,
        content_type: str = ...,
        encoding: str = ...,
        proto: Protocol = ...,
        allow_pickle: bool = ...
    ) -> Model: ...
    @classmethod
    def parse_file(
        cls: Type["BaseModel"],
        path: Union[str, Path],
        *,
        content_type: str = ...,
        encoding: str = ...,
        proto: Protocol = ...,
        allow_pickle: bool = ...
    ) -> Model: ...
    @classmethod
    def from_orm(cls: Type["BaseModel"], obj: Any) -> BaseModel: ...
    @classmethod
    def construct(
        cls: Type["BaseModel"], _fields_set: Optional[SetStr] = ..., **values: Any
    ) -> BaseModel:
        """
        Creates a new model setting __dict__ and __fields_set__ from trusted or pre-validated data.
        Default values are respected, but no other validation is performed.
        Behaves as if `Config.extra = 'allow'` was set since it adds all passed values
        """
        ...
    def copy(
        self: BaseModel,
        *,
        include: Union[AbstractSetIntStr, MappingIntStrAny] = ...,
        exclude: Union[AbstractSetIntStr, MappingIntStrAny] = ...,
        update: DictStrAny = ...,
        deep: bool = ...
    ) -> BaseModel:
        """
        Duplicate a model, optionally choose which fields to include, exclude and change.

        :param include: fields to include in new model
        :param exclude: fields to exclude from new model, as with values this takes precedence over include
        :param update: values to change/add in the new model. Note: the data is not validated before creating
            the new model: you should trust this data
        :param deep: set to `True` to make a deep copy of the model
        :return: new model instance
        """
        ...
    @classmethod
    def schema(cls, by_alias: bool = ..., ref_template: str = ...) -> DictStrAny: ...
    @classmethod
    def schema_json(
        cls, *, by_alias: bool = ..., ref_template: str = ..., **dumps_kwargs: Any
    ) -> str: ...
    @classmethod
    def __get_validators__(cls) -> CallableGenerator: ...
    @classmethod
    def validate(cls: Type["BaseModel"], value: Any) -> BaseModel: ...
    @classmethod
    def update_forward_refs(cls, **localns: Any) -> None:
        """
        Try to update ForwardRefs on fields based on this Model, globalns and localns.
        """
        ...
    def __iter__(self) -> TupleGenerator:
        """
        so `dict(model)` works
        """
        ...
    def __eq__(self, other: Any) -> bool: ...
    def __repr_args__(self) -> ReprArgs: ...

_is_base_model_class_defined = ...

def create_model(
    __model_name: str,
    *,
    __config__: Type[BaseConfig] = ...,
    __base__: Type["BaseModel"] = ...,
    __module__: str = ...,
    __validators__: Dict[str, classmethod] = ...,
    **field_definitions: Any
) -> Type["BaseModel"]:
    """
    Dynamically create a model.
    :param __model_name: name of the created model
    :param __config__: config class to use for the new model
    :param __base__: base class for the new model to inherit from
    :param __module__: module of the created model
    :param __validators__: a dict of method names and @validator class methods
    :param field_definitions: fields of the model (or extra fields if a base is supplied)
        in the format `<name>=(<type>, <default default>)` or `<name>=<default value>, e.g.
        `foobar=(str, ...)` or `foobar=123`, or, for complex use-cases, in the format
        `<name>=<FieldInfo>`, e.g. `foo=Field(default_factory=datetime.utcnow, alias='bar')`
    """
    ...

_missing = ...

def validate_model(
    model: Type[BaseModel], input_data: DictStrAny, cls: ModelOrDc = ...
) -> Tuple[DictStrAny, SetStr, Optional[ValidationError]]:
    """
    validate data against a model.
    """
    ...
