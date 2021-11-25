
from types import BuiltinFunctionType, FunctionType
from typing import Any, Callable, Dict, List
from typing import Literal as LiteralType
from typing import Optional as OptionalType
from typing import Tuple, Union, overload

GenericType = Union["Schema", "And", "Or", "Use", "Optional", "Regex", "Literal", "Const"]
AcceptedDictType = Dict[Union[str, GenericType], Any]
_CallableLike = Union[FunctionType, BuiltinFunctionType, Callable[..., Any]]

class SchemaError(Exception):
    """Error during Schema validation."""
    @overload
    def __init__(self, autos: str, errors: str=...) -> None:
        ...
    @overload
    def __init__(self, autos: str, errors: List[str]=...) -> None:
        ...
    @overload
    def __init__(self, autos: List[str], errors: None=...) -> None:
        ...

    @property
    def code(self) -> str:
        """
        Removes duplicates values in auto and error list.
        parameters.
        """
        ...


class SchemaWrongKeyError(SchemaError): ...


class SchemaMissingKeyError(SchemaError): ...


class SchemaOnlyOneAllowedError(SchemaError): ...


class SchemaForbiddenKeyError(SchemaError): ...


class SchemaUnexpectedTypeError(SchemaError): ...


class And:
    """
    Utility function to combine validation directives in AND Boolean fashion.
    """
    @overload
    def __init__(self, *args: _CallableLike, error: List[str]=...,schema: "Schema"=...,ignore_extra_keys: bool=...) -> None:
        ...
    @overload
    def __init__(self, *args: _CallableLike, error: List[str]=...,schema: None=..., ignore_extra_keys: bool=...) -> None:
        ...

    def __repr__(self) -> str:
        ...

    @property
    def args(self) -> Tuple[_CallableLike, ...]: ...

    def validate(self, data: Any) -> Any: ...



class Or(And):
    """Utility function to combine validation directives in a OR Boolean
    fashion."""
    @overload
    def __init__(self, *args: _CallableLike, error: List[str]=...,schema: None=..., ignore_extra_keys: bool=..., only_one: bool=...) -> None:
        ...
    @overload
    def __init__(self, *args: _CallableLike, error: List[str]=...,schema: "Schema"=..., ignore_extra_keys: bool=..., only_one: bool=...) -> None:
        ...

    def reset(self) -> None:
        ...

    def validate(self, data: Any) -> Any:
        ...


class Regex:
    NAMES: List[str]= ...
    def __init__(self, pattern_str: str, flags: OptionalType[int]=..., error: OptionalType[str]=...) -> None:
        ...

    def __repr__(self) -> str:
        ...

    @property
    def pattern_str(self) -> str:
        ...

    def validate(self, data: str) -> str:
        ...


class Use:
    def __init__(self, callable_: _CallableLike, error: OptionalType[str]=...) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def validate(self, data: Any) -> Any:
        ...


COMPARABLE=LiteralType["0"]
CALLABLE=LiteralType["1"]
VALIDATOR=LiteralType["2"]
TYPE=LiteralType["3"]
DICT = LiteralType["4"]
ITERABLE=LiteralType["5"]

def _priority(s: object) -> Union[CALLABLE, COMPARABLE, VALIDATOR, TYPE, DICT, ITERABLE]: ...

def _invoke_with_optional_kwargs(f: Callable[..., Any], **kwargs: Any) -> Any: ...

class Schema:
    """
    Entry point of the library, use this class to instantiate validation
    schema for the data that will be validated.
    """
    def __init__(self, schema: Union[Schema, AcceptedDictType], error: OptionalType[str]=..., ignore_extra_keys: OptionalType[bool]=..., name: OptionalType[str]=..., description: OptionalType[str]=..., as_reference: bool=...)-> None:
        ...

    def __repr__(self) -> str:
        ...

    @property
    def schema(self) -> AcceptedDictType:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def ignore_extra_keys(self) -> bool:
        ...

    @staticmethod
    def _dict_key_priority(s: Union[GenericType, object]) -> Union[float, int]: ...

    @staticmethod
    def _is_optional_type(s: Union[GenericType, object]) -> bool:...

    def is_valid(self, data: Any, **kwargs: Any) -> bool:
        """Return whether the given data has passed all the validations
        that were specified in the given schema.
        """
        ...

    def _prepend_schema_name(self, message: str) -> str:...

    @overload
    def validate(self, data: AcceptedDictType) -> AcceptedDictType:
        ...
    @overload
    def validate(self, data: Any) -> Any:
        ...

    def json_schema(self, schema_id: Any, use_refs: bool=...) -> Dict[str, Any]:
        """Generate a draft-07 JSON schema dict representing the Schema.
        This method must be called with a schema_id.

        :param schema_id: The value of the $id on the main schema
        :param use_refs: Enable reusing object references in the resulting JSON schema.
                         Schemas with references are harder to read by humans, but are a lot smaller when there
                         is a lot of reuse
        """
        ...



class Optional(Schema):
    """Marker for an optional part of the validation Schema."""
    _MARKER: object = ...
    default: object
    key: str
    @overload
    def __init__(self, schema: Union[Schema, AcceptedDictType], error: Union[str, List[str]]=..., ignore_extra_keys: OptionalType[bool]=..., name: OptionalType[str]=..., description: OptionalType[str]=..., as_reference: OptionalType[bool]=..., default: OptionalType[Any]=...) -> None:
        ...
    @overload
    def __init__(self, schema: str, error: Union[str, List[str]]=..., ignore_extra_keys: OptionalType[bool]=..., name: OptionalType[str]=..., description: OptionalType[str]=..., as_reference: OptionalType[bool]=..., default: OptionalType[Any]=...) -> None:
        ...

    def __hash__(self) -> int:
        ...

    def __eq__(self, other: Optional) -> bool:
        ...

    def reset(self) -> None:
        ...


_HookCallback = Callable[[str, AcceptedDictType, str], SchemaError]

class Hook(Schema):
    key: Schema
    def __init__(self, schema: Union[Schema, AcceptedDictType], error: OptionalType[str]=..., ignore_extra_keys: OptionalType[bool]=..., name: OptionalType[str]=..., description: OptionalType[str]=..., as_reference: OptionalType[bool]=..., handler: OptionalType[_HookCallback]=...) -> None:
        ...



class Forbidden(Hook):
    handler: Callable[[str, AcceptedDictType, str], SchemaForbiddenKeyError]
    def __init__(self, schema: AcceptedDictType, error: OptionalType[str] = ..., ignore_extra_keys: OptionalType[bool] = ..., name: OptionalType[str] = ..., description: OptionalType[str] = ..., as_reference: OptionalType[bool] = ...) -> None:
        ...

    @staticmethod
    def _default_function(nkey: str, data: Any, error: Exception) -> None:...


class Literal(object):
    def __init__(self, value: str, description: OptionalType[str]=...) -> None:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...

    @property
    def schema(self) -> str:
        ...


class Const(Schema):
    def validate(self, data: Union[Schema, AcceptedDictType]) -> AcceptedDictType:
        ...

def _callable_str(callable_: Callable[..., Any]) -> str: ...

def _plural_s(sized: str) -> str: ...
