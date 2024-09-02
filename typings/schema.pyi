import typing as t
from types import BuiltinFunctionType
from types import FunctionType
from typing import overload

class SchemaError(Exception):
    def __init__(
        self, autos: str | list[str], errors: str | list[str] | None = ...
    ) -> None: ...
    @property
    def code(self) -> str: ...

class SchemaWrongKeyError(SchemaError): ...
class SchemaMissingKeyError(SchemaError): ...
class SchemaOnlyOneAllowedError(SchemaError): ...
class SchemaForbiddenKeyError(SchemaError): ...
class SchemaUnexpectedTypeError(SchemaError): ...

class OpsMeta(t.Protocol):
    @overload
    def validate(self, data: AcceptedDictType, **kwargs: t.Any) -> AcceptedDictType: ...
    @overload
    def validate(self, data: t.Any, **kwargs: t.Any) -> t.Any: ...

class And(OpsMeta):
    def __init__(
        self,
        *args: _SchemaLike,
        error: list[str] = ...,
        schema: Schema | None = ...,
        ignore_extra_keys: bool = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def args(self) -> tuple[_CallableLike, ...]: ...

class Or(And):
    def __init__(
        self,
        *args: _SchemaLike | t.MutableSequence[t.Any] | None,
        error: list[str] = ...,
        schema: None = ...,
        ignore_extra_keys: bool = ...,
        only_one: bool = ...,
    ) -> None: ...
    def reset(self) -> None: ...

class Regex(OpsMeta):
    NAMES: list[str] = ...

    def __init__(
        self,
        pattern_str: str,
        flags: int | None = ...,
        error: str | None = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def pattern_str(self) -> str: ...

class Use(OpsMeta):
    def __init__(self, callable_: _CallableLike, error: str | None = ...) -> None: ...
    def __repr__(self) -> str: ...

COMPARABLE = t.Literal["0"]
CALLABLE = t.Literal["1"]
VALIDATOR = t.Literal["2"]
TYPE = t.Literal["3"]
DICT = t.Literal["4"]
ITERABLE = t.Literal["5"]

def _priority(
    s: OpsType,
) -> CALLABLE | COMPARABLE | VALIDATOR | TYPE | DICT | ITERABLE: ...
def _invoke_with_optional_kwargs(
    f: t.Callable[..., t.Any], **kwargs: t.Any
) -> t.Any: ...

class Schema(OpsMeta):
    def __init__(
        self,
        schema: _SchemaLike | type | AcceptedDictType,
        error: str | None = ...,
        ignore_extra_keys: bool | None = ...,
        name: str | None = ...,
        description: str | None = ...,
        as_reference: bool = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def schema(self) -> AcceptedDictType: ...
    @property
    def description(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def ignore_extra_keys(self) -> bool: ...
    @staticmethod
    def _dict_key_priority(s: OpsType) -> float | int: ...
    @staticmethod
    def _is_optional_type(s: OpsType) -> bool: ...
    def is_valid(self, data: t.Any, **kwargs: t.Any) -> bool: ...
    def _prepend_schema_name(self, message: str) -> str: ...
    def json_schema(self, schema_id: str, use_refs: bool = ...) -> dict[str, t.Any]: ...

class Optional(Schema):
    _MARKER: object = ...
    default: object
    key: str

    def __init__(
        self,
        schema: _SchemaLike | type | str,
        error: str | list[str] | None = ...,
        ignore_extra_keys: bool | None = ...,
        name: str | None = ...,
        description: str | None = ...,
        as_reference: bool | None = ...,
        default: t.Any = ...,
    ) -> None: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: Optional) -> bool: ...
    def reset(self) -> None: ...

_HookCallback = t.Callable[[str, t.Any, str | list[str]], SchemaError | None | t.Any]

class Hook(Schema):
    key: Schema

    def __init__(
        self,
        schema: _SchemaLike | str,
        error: str | list[str] | None = ...,
        ignore_extra_keys: bool | None = ...,
        name: str | None = ...,
        description: str | None = ...,
        as_reference: bool | None = ...,
        handler: _HookCallback | None = ...,
    ) -> None: ...

class Forbidden(Hook): ...

class Literal:
    def __init__(self, value: str, description: str | None = ...) -> None: ...
    @property
    def description(self) -> str: ...
    @property
    def schema(self) -> str: ...

class Const(Schema): ...

OpsType = Schema | And | Or | Use | Optional | Regex | Literal | Const
AcceptedDictType = dict[str | OpsType, t.Any]
_CallableLike = FunctionType | BuiltinFunctionType | t.Callable[..., t.Any]
_SchemaLike = _CallableLike | OpsType | AcceptedDictType
