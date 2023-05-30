from __future__ import annotations

import re
import typing as t
import inspect
import warnings
import dataclasses
from enum import Enum
from uuid import UUID
from types import GeneratorType
from decimal import Decimal
from pathlib import Path
from datetime import date
from datetime import time
from datetime import datetime
from datetime import timedelta
from functools import lru_cache
from ipaddress import IPv4Address
from ipaddress import IPv4Network
from ipaddress import IPv6Address
from ipaddress import IPv6Network
from ipaddress import IPv4Interface
from ipaddress import IPv6Interface
from collections import deque
from collections import defaultdict

import attr
from typing_extensions import Literal

from ...types import get_args
from ...types import LazyType
from ...types import get_origin
from ...types import is_none_type
from ...types import is_namedtuple
from ...types import is_literal_type
from ...types import is_callable_type
from ...types import all_literal_values
from ...types import lenient_issubclass
from ...types import lenient_resolve_types
from ...utils import LazyLoader
from ...utils import bentoml_cattr
from ...utils.pkg import pkg_version_info
from ....exceptions import NotFound
from ....exceptions import InvalidArgument
from ....exceptions import BentoMLException
from ....exceptions import InternalServerError
from .specification import Schema

if t.TYPE_CHECKING:
    import pydantic
    import pydantic.schema as schema
    from attrs import AttrsInstance

    if pkg_version_info("pydantic")[0] >= 2:
        import pydantic.json_schema as jschema

    TypeAttrsOrEnum = type[AttrsInstance] | type[Enum]
    TypeAttrsSet = set[TypeAttrsOrEnum]

else:
    _exc_msg = (
        "Missing required dependency: 'pydantic'. Install with 'pip install pydantic'."
    )
    pydantic = LazyLoader("pydantic", globals(), "pydantic", exc_msg=_exc_msg)
    schema = LazyLoader("schema", globals(), "pydantic.schema", exc_msg=_exc_msg)
    jschema = LazyLoader(
        "jschema", globals(), "pydantic.json_schema", exc_msg="Pydantic v2 is required."
    )

REF_PREFIX = "#/components/schemas/"


class SkipField(BentoMLException):
    """Raises when a field is a callable and not able to convert to JSON schema."""


# NOTE: attrs to json schema


def isoformat(o: date | time) -> str:
    return o.isoformat()


def decimal_encoder(dec_value: Decimal) -> int | float:
    """
    Vendorred from pydantic v1

    Encodes a Decimal as int of there's no exponent, otherwise float

    This is useful when we use ConstrainedDecimal to represent Numeric(x,0)
    where a integer (but not int typed) is used. Encoding this as a float
    results in failed round-tripping between encode and parse.
    Our Id type is a prime example of this.

    >>> decimal_encoder(Decimal("1.0"))
    1.0

    >>> decimal_encoder(Decimal("1"))j
    1
    """
    # NOTE: Hmm, pyright recognize it as a Literal['n', 'N', 'F'], but when tested exponent return int?
    if t.cast(int, dec_value.as_tuple().exponent) >= 0:
        return int(dec_value)
    else:
        return float(dec_value)


ENCODERS_BY_TYPE: dict[type[t.Any], t.Callable[[t.Any], t.Any]] = {
    bytes: lambda o: o.decode(),
    date: isoformat,
    datetime: isoformat,
    time: isoformat,
    timedelta: lambda td: td.total_seconds(),
    Decimal: decimal_encoder,
    Enum: lambda o: o.value,
    frozenset: list,
    deque: list,
    GeneratorType: list,
    IPv4Address: str,
    IPv4Interface: str,
    IPv4Network: str,
    IPv6Address: str,
    IPv6Interface: str,
    IPv6Network: str,
    Path: str,
    t.Pattern: lambda o: o.pattern,
    set: list,
    UUID: str,
}


def attrs_encoder(o: t.Any) -> t.Any:
    if LazyType["pydantic.BaseModel"]("pydantic.BaseModel").isinstance(o):
        if pkg_version_info("pydantic")[0] >= 2:
            return o.model_dump()
        else:
            obj_dict = o.dict()
            if "__root__" in obj_dict:
                obj_dict = obj_dict.get("__root__")
            return obj_dict
    elif dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    elif attr.has(o):
        return bentoml_cattr.unstructure(o)

    for base in o.__class__.__mro__[:-1]:
        try:
            encoder = ENCODERS_BY_TYPE[base]
        except KeyError:
            continue
        return encoder(o)
    else:
        raise TypeError(
            f"Object of type '{o.__class__.__name__}' is not JSON serializable"
        )


def encode_default(o: t.Any) -> t.Any:
    if (
        LazyType["pydantic.BaseModel"]("pydantic.BaseModel").isinstance(o)
        or dataclasses.is_dataclass(o)
        or attr.has(o)
    ):
        o = t.cast("dict[str, t.Any]", attrs_encoder(o))

    if LazyType[DictStrAny](dict).isinstance(o):
        return {encode_default(k): encode_default(v) for k, v in o.items()}
    elif isinstance(o, Enum):
        return o.value
    elif isinstance(o, (int, float, str)):
        return o
    elif LazyType[ListAny](list).isinstance(o) or LazyType[TupleAny](tuple).isinstance(
        o
    ):
        tp = o.__class__
        seq_args = (encode_default(i) for i in o)
        return tp(*seq_args) if is_namedtuple(tp) else tp(seq_args)
    elif o is None:
        return None
    else:
        return attrs_encoder(o)


def get_long_model_name(model: TypeAttrsOrEnum) -> str:
    return f"{model.__module__}__{model.__qualname__}".replace(".", "__")


def normalize_name(name: str) -> str:
    """
    Normalizes the given name. This can be applied to either a model *or* enum.
    """
    return re.sub(r"[^a-zA-Z0-9.\-_]", "_", name)


def get_flat_attrs_from_attr_model(
    attrs_model: type[AttrsInstance], known_attrs: TypeAttrsSet | None = None
) -> TypeAttrsSet:
    known_attrs = known_attrs or set()
    flat_attrs: TypeAttrsSet = {attrs_model}
    known_attrs.add(attrs_model)
    flat_attrs |= get_flat_attrs_from_fields(
        attr.fields(attrs_model), known_attrs=known_attrs
    )
    return flat_attrs


def get_flat_attrs_from_fields(
    fields: tuple[attr.Attribute[t.Any]], known_attrs: TypeAttrsSet
) -> TypeAttrsSet:
    flat_models: TypeAttrsSet = set()
    for field in fields:
        flat_models |= get_flat_attrs_from_field(field, known_attrs=known_attrs)
    return flat_models


def get_flat_attrs_from_field(
    field: attr.Attribute[t.Any], known_attrs: TypeAttrsSet
) -> TypeAttrsSet:
    flat_models: TypeAttrsSet = set()
    field_type = field.type
    if field_type and hasattr(field_type, "type") and attr.has(field_type.type):
        field_type = field_type.type

    if field_type and attr.has(field_type) and field.type not in known_attrs:
        flat_models |= get_flat_attrs_from_attr_model(
            field_type, known_attrs=known_attrs
        )
    elif field_type and lenient_issubclass(field_type, Enum):
        flat_models.add(field_type)
    return flat_models


def get_attrs_name_map(
    unique_attrs: TypeAttrsSet,
) -> dict[TypeAttrsOrEnum, str]:
    name_model_map: dict[str, TypeAttrsOrEnum] = {}
    conflicts: set[str] = set()
    for attr_model in unique_attrs:
        attr_name = normalize_name(attr_model.__name__)
        if attr_name in conflicts:
            attr_name = get_long_model_name(attr_model)
            name_model_map[attr_name] = attr_model
        elif attr_name in name_model_map:
            conflicts.add(attr_name)
            conflict_attrs = name_model_map.pop(attr_name)
            name_model_map[attr_name] = attr_model
            name_model_map[get_long_model_name(conflict_attrs)] = conflict_attrs
            name_model_map[get_long_model_name(attr_model)] = attr_model
        else:
            name_model_map[attr_name] = attr_model
    return {v: k for k, v in name_model_map.items()}


def attrs_process_schema(
    attr_model: type[AttrsInstance],
    *,
    by_alias: bool = True,
    attrs_name_map: dict[TypeAttrsOrEnum, str] | None = None,
    ref_prefix: str | None = REF_PREFIX,
    known_attrs: TypeAttrsSet | None = None,
) -> tuple[dict[str, t.Any], dict[str, t.Any], set[str]]:
    known_attrs = known_attrs or set()

    if lenient_issubclass(attr_model, Enum):
        return enum_process_schema(t.cast("type[Enum]", attr_model)), {}, set()

    if attrs_name_map is None:
        attrs_name_map = get_attrs_name_map(
            get_flat_attrs_from_attr_model(attr_model, known_attrs=known_attrs)
        )

    s = {"title": attr_model.__name__}
    docstring = inspect.getdoc(s)
    if docstring:
        s["description"] = docstring

    known_attrs.add(attr_model)
    m_schema, m_definitions, nested_attrs = attrs_type_schema(
        attr_model,
        by_alias=by_alias,
        attrs_name_map=attrs_name_map,
        ref_prefix=ref_prefix,
        known_attrs=known_attrs,
    )
    s.update(m_schema)
    return s, m_definitions, nested_attrs


def attrs_type_schema(
    attr_model: type[AttrsInstance],
    *,
    by_alias: bool = True,
    attrs_name_map: dict[TypeAttrsOrEnum, str],
    ref_prefix: str | None = REF_PREFIX,
    known_attrs: TypeAttrsSet,
) -> tuple[dict[str, t.Any], dict[str, t.Any], set[str]]:
    properties: dict[str, t.Any] = {}
    required: list[t.Any] = []
    definitions: dict[str, t.Any] = {}
    nested_models: set[str] = set()

    for attribute in attr.fields(attr_model):
        try:
            f_schema, f_definitions, f_nested_models = attrs_attribute_schema(
                attribute,
                by_alias=by_alias,
                attrs_name_map=attrs_name_map,
                ref_prefix=ref_prefix,
                known_attrs=known_attrs,
            )
        except SkipField as skip:
            warnings.warn(skip.message, UserWarning)
            continue

        definitions.update(f_definitions)
        nested_models.update(f_nested_models)

        if by_alias:
            properties[attribute.alias] = f_schema
            if attribute.default is attr.NOTHING:
                required.append(attribute.alias)
        else:
            properties[attribute.name] = f_schema
            if attribute.default is attr.NOTHING:
                required.append(attribute.name)

    out_schema: dict[str, t.Any] = {"type": "object", "properties": properties}
    if required:
        out_schema["required"] = required
    return out_schema, definitions, nested_models


def get_field_info_schema(
    attribute: attr.Attribute[t.Any],
    *,
    schema_overrides: bool = False,
    by_alias: bool = True,
) -> tuple[dict[str, t.Any], bool]:
    schema_: dict[str, t.Any] = {}

    if attribute.type and not lenient_issubclass(attribute.type, Enum):
        schema_["title"] = attribute.alias if by_alias else attribute.name
    if (
        attribute.default is not attr.NOTHING
        and attribute.type
        and not is_callable_type(attribute.type)
    ):
        schema_["default"] = encode_default(attribute.default)
        schema_overrides = True

    return schema_, schema_overrides


def attrs_attribute_schema(
    attribute: attr.Attribute[t.Any],
    *,
    by_alias: bool = True,
    attrs_name_map: dict[TypeAttrsOrEnum, str],
    ref_prefix: str | None = REF_PREFIX,
    known_attrs: TypeAttrsSet | None = None,
) -> tuple[dict[str, t.Any], dict[str, t.Any], set[str]]:
    schema_, schema_overrides = get_field_info_schema(attribute, by_alias=by_alias)

    # TODO: maybe we can support validation schema here?

    f_schema, f_definitions, f_nested_models = attrs_attribute_type_schema(
        attribute,
        by_alias=by_alias,
        attrs_name_map=attrs_name_map,
        schema_overrides=schema_overrides,
        ref_prefix=ref_prefix,
        known_attrs=known_attrs or set(),
    )

    # $ref will only be returned when there are no schema_overrides
    if "$ref" in f_schema:
        return f_schema, f_definitions, f_nested_models
    else:
        schema_.update(f_schema)
        return schema_, f_definitions, f_nested_models


def attrs_attribute_type_schema(
    attribute: attr.Attribute[t.Any],
    *,
    by_alias: bool = True,
    attrs_name_map: dict[TypeAttrsOrEnum, str],
    schema_overrides: bool,
    ref_prefix: str | None = REF_PREFIX,
    known_attrs: TypeAttrsSet,
) -> tuple[dict[str, t.Any], dict[str, t.Any], set[str]]:
    definitions: dict[str, t.Any] = {}
    nested_attrs: set[str] = set()
    attrs_schema: dict[str, t.Any]

    # TODO: we need more complex type handling here, similar to what pydantic does.
    if hasattr(attribute, "type") and get_origin(attribute.type) in {
        list,
        tuple,
        t.Sequence,
        set,
        frozenset,
        t.Iterable,
        deque,
    }:
        (
            items_schema,
            items_definitions,
            items_nested_models,
        ) = attrs_attribute_singleton_schema(
            attribute,
            by_alias=by_alias,
            attrs_name_map=attrs_name_map,
            schema_overrides=schema_overrides,
            ref_prefix=ref_prefix,
            known_attrs=known_attrs,
        )
        definitions.update(items_definitions)
        nested_attrs.update(items_nested_models)
        attrs_schema = {"type": "array", "items": items_schema}
        if attribute.type in {t.Set, t.FrozenSet}:
            attrs_schema["uniqueItems"] = True
    else:
        attrs_schema, f_definitions, f_nested_models = attrs_attribute_singleton_schema(
            attribute,
            by_alias=by_alias,
            attrs_name_map=attrs_name_map,
            schema_overrides=schema_overrides,
            ref_prefix=ref_prefix,
            known_attrs=known_attrs,
        )
        definitions.update(f_definitions)
        nested_attrs.update(f_nested_models)

    return attrs_schema, definitions, nested_attrs


def attrs_attribute_singleton_schema(
    attribute: attr.Attribute[t.Any],
    *,
    by_alias: bool = True,
    schema_overrides: bool = False,
    attrs_name_map: dict[TypeAttrsOrEnum, str],
    ref_prefix: str | None = REF_PREFIX,
    known_attrs: TypeAttrsSet,
) -> tuple[dict[str, t.Any], dict[str, t.Any], set[str]]:
    """Used by attrs_attribute_type_schema, probably should use that function instead."""
    definitions: dict[str, t.Any] = {}
    nested_models: set[str] = set()

    # This branch is an early hit when attribute is a generic type when handling container type
    # Hmm, maybe we should handling this in attrs_attribute_type_schema?
    # XXX: This is probably a bug, but it works for now.
    if not hasattr(attribute, "type"):
        schema_: dict[str, t.Any] = {}
        add_field_type_to_schema(attribute, schema_)
        return schema_, definitions, nested_models

    # NOTE: resolve attribute.type if it is a string type, possibly a forward reference
    tp = lenient_resolve_types(attribute.type)

    if get_origin(tp) in {list, tuple, t.Sequence, set, frozenset, t.Iterable, deque}:
        return attrs_attribute_singleton_container_schema(
            attribute,
            by_alias=by_alias,
            attrs_name_map=attrs_name_map,
            schema_overrides=schema_overrides,
            ref_prefix=ref_prefix,
            known_attrs=known_attrs,
        )
    # NOTE: early out some types
    if (
        tp is t.Any
        or tp is object
        or tp.__class__ == t.TypeVar
        or get_origin(tp) is type
    ):
        # NOTE: no restriction for generic type
        return {}, definitions, nested_models
    if is_none_type(tp):
        return {"type": "null"}, definitions, nested_models
    assert tp is not None  # NOTE: none type should already be handled by now
    if is_callable_type(tp):
        raise SkipField(
            f"Callable type {tp} is excluded from JSON schema since JSON schema has no equivalent type."
        )

    f_schema: dict[str, t.Any] = {}
    if attribute.default is not attr.NOTHING:
        f_schema["const"] = attribute.default

    if is_literal_type(tp):
        values = tuple(
            x.value if isinstance(x, Enum) else x for x in all_literal_values(tp)
        )

        if len({v.__class__ for v in values}) > 1:
            return attrs_attribute_schema(
                multitypes_literal_field_for_schema(values, attribute),
                by_alias=by_alias,
                attrs_name_map=attrs_name_map,
                ref_prefix=ref_prefix,
                known_attrs=known_attrs,
            )

        # All values have the same type
        tp = values[0].__class__
        f_schema["enum"] = list(values)
        add_field_type_to_schema(tp, f_schema)
    elif lenient_issubclass(tp, Enum):
        enum_name = attrs_name_map[tp]
        f_schema, schema_overrides = get_field_info_schema(
            attribute, schema_overrides=schema_overrides, by_alias=by_alias
        )
        f_schema.update(get_schema_ref(enum_name, ref_prefix, schema_overrides))
        definitions[enum_name] = enum_process_schema(tp)
    elif is_namedtuple(tp):
        sub_schema, *_ = attrs_process_schema(
            attr.fields(tp),
            by_alias=by_alias,
            attrs_name_map=attrs_name_map,
            ref_prefix=ref_prefix,
            known_attrs=known_attrs,
        )
        items_schemas = list(sub_schema["properties"].values())
        f_schema.update(
            {
                "type": "array",
                "items": items_schemas,
                "minItems": len(items_schemas),
                "maxItems": len(items_schemas),
            }
        )
    elif not attr.has(tp):
        add_field_type_to_schema(tp, f_schema)

    if f_schema:
        return f_schema, definitions, nested_models

    # TODO: handle dataclass-based attrs class

    # NOTE: finally handle the nested attrs class
    if attr.has(tp):
        model_name = attrs_name_map[tp]
        if tp not in known_attrs:
            sub_schema, sub_definitions, sub_nested_models = attrs_process_schema(
                tp,
                by_alias=by_alias,
                attrs_name_map=attrs_name_map,
                ref_prefix=ref_prefix,
                known_attrs=known_attrs,
            )
            definitions.update(sub_definitions)
            definitions[model_name] = sub_schema
            nested_models.update(sub_nested_models)
        else:
            nested_models.add(model_name)
        schema_ref = get_schema_ref(model_name, ref_prefix, schema_overrides)
        return schema_ref, definitions, nested_models

    # For generics with no args
    args = get_args(tp)
    if args is not None and not args and t.Generic in tp.__bases__:
        return f_schema, definitions, nested_models

    raise ValueError(f"Value not declarable with JSON Schema, field: {attribute}")


def attrs_attribute_singleton_container_schema(
    field: attr.Attribute[t.Any],
    *,
    by_alias: bool = True,
    schema_overrides: bool = False,
    attrs_name_map: dict[TypeAttrsOrEnum, str],
    ref_prefix: str | None = REF_PREFIX,
    known_attrs: TypeAttrsSet,
) -> tuple[dict[str, t.Any], dict[str, t.Any], set[str]]:
    args = get_args(field.type)
    definitions: dict[str, t.Any] = {}
    nested_models: set[str] = set()

    if len(args) == 1:
        return attrs_attribute_type_schema(
            args[0],
            by_alias=by_alias,
            schema_overrides=schema_overrides,
            attrs_name_map=attrs_name_map,
            ref_prefix=ref_prefix,
            known_attrs=known_attrs,
        )
    else:
        s: dict[str, t.Any] = {}
        # TODO: support https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#discriminator-object

        sub_field_schemas: list[dict[str, t.Any]] = []
        for sf in args:
            (
                sub_schema,
                sub_definitions,
                sub_nested_models,
            ) = attrs_attribute_type_schema(
                sf,
                by_alias=by_alias,
                attrs_name_map=attrs_name_map,
                schema_overrides=schema_overrides,
                ref_prefix=ref_prefix,
                known_attrs=known_attrs,
            )
            definitions.update(sub_definitions)
            if schema_overrides and "allOf" in sub_schema:
                # if the sub_field is a referenced schema we only need the referenced
                # object. Otherwise we will end up with several allOf inside anyOf/oneOf.
                sub_schema = sub_schema["allOf"][0]

            # if sub_schema.keys() == {'discriminator', 'oneOf'}:
            #     # we don't want discriminator information inside oneOf choices, this is dealt with elsewhere
            #     sub_schema.pop('discriminator')
            sub_field_schemas.append(sub_schema)
            nested_models.update(sub_nested_models)
        # NOTE: uncomment below when discriminator is supported
        # s['oneOf' if field_has_discriminator else 'anyOf'] = sub_field_schemas
        s["anyOf"] = sub_field_schemas
        return s, definitions, nested_models


def multitypes_literal_field_for_schema(
    values: tuple[t.Any, ...], field: attr.Attribute[t.LiteralString]
) -> attr.Attribute[t.Any]:
    """
    To support `Literal` with values of different types, we split it into multiple `Literal` with same type
    e.g. `Literal['qwe', 'asd', 1, 2]` becomes `Union[Literal['qwe', 'asd'], Literal[1, 2]]`
    """
    literal_distinct_types: defaultdict[str, list[t.LiteralString]] = defaultdict(list)
    for v in values:
        literal_distinct_types[v.__class__].append(v)

    distinct_literals: t.Generator[t.Any, None, None] = (
        Literal[tuple(same_type_values)]  # type: ignore
        for same_type_values in literal_distinct_types.values()
    )

    return field.evolve(type=t.Union[tuple(distinct_literals)])  # type: ignore


def get_schema_ref(
    name: str, ref_prefix: str | None, schema_overrides: bool
) -> dict[str, t.Any]:
    if ref_prefix is None:
        ref_prefix = REF_PREFIX

    # TODO: support ref_template
    schema_ref = {"$ref": ref_prefix + name}

    return {"allOf": [schema_ref]} if schema_overrides else schema_ref


if t.TYPE_CHECKING:
    DictStrAny = dict[str, t.Any]
    ListAny = list[t.Any]
    TupleAny = tuple[t.Any, ...]
    SetAny = set[t.Any]
    PatternAny = t.Pattern[t.Any]
    FrozenSetAny = frozenset[t.Any]
else:
    DictStrAny = dict
    ListAny = list
    TupleAny = tuple
    SetAny = set
    PatternAny = t.Pattern
    FrozenSetAny = frozenset

# Order is important, e.g. subclasses of str must go before str
# This is used only for standard library types
# NOTE: Vendorred from pydantic v1 logics
FIELD_CLASS_TO_SCHEMA: tuple[tuple[t.Any, dict[str, t.Any]], ...] = (
    (Path, {"type": "string", "format": "path"}),
    (datetime, {"type": "string", "format": "date-time"}),
    (date, {"type": "string", "format": "date"}),
    (time, {"type": "string", "format": "time"}),
    (timedelta, {"type": "number", "format": "time-delta"}),
    (IPv4Network, {"type": "string", "format": "ipv4network"}),
    (IPv6Network, {"type": "string", "format": "ipv6network"}),
    (IPv4Interface, {"type": "string", "format": "ipv4interface"}),
    (IPv6Interface, {"type": "string", "format": "ipv6interface"}),
    (IPv4Address, {"type": "string", "format": "ipv4"}),
    (IPv6Address, {"type": "string", "format": "ipv6"}),
    (PatternAny, {"type": "string", "format": "regex"}),
    (str, {"type": "string"}),
    (bytes, {"type": "string", "format": "binary"}),
    (bool, {"type": "boolean"}),
    (int, {"type": "integer"}),
    (float, {"type": "number"}),
    (Decimal, {"type": "number"}),
    (UUID, {"type": "string", "format": "uuid"}),
    (DictStrAny, {"type": "object"}),
    (ListAny, {"type": "array", "items": {}}),
    (TupleAny, {"type": "array", "items": {}}),
    (SetAny, {"type": "array", "items": {}, "uniqueItems": True}),
    (FrozenSetAny, {"type": "array", "items": {}, "uniqueItems": True}),
)


def enum_process_schema(enum: type[Enum]) -> dict[str, t.Any]:
    # similar to enum_process_schema in pydantic v1
    import inspect

    schema_: dict[str, t.Any] = {
        "title": enum.__name__,
        # Python assigns all enums a default docstring value of 'An enumeration', so
        # all enums will have a description field even if not explicitly provided.
        "description": inspect.cleandoc(enum.__doc__ or "An enumeration."),
        # Add enum values and the enum field type to the schema.
        "enum": [item.value for item in t.cast(t.Iterable[Enum], enum)],
    }

    add_field_type_to_schema(enum, schema_)

    return schema_


def add_field_type_to_schema(field_type: t.Any, schema_: dict[str, t.Any]) -> None:
    # Update the schema with the field type
    for type_, t_schema in FIELD_CLASS_TO_SCHEMA:
        # Fallback for `typing.Pattern` and `re.Pattern` as they are not a valid class
        if lenient_issubclass(field_type, type_) or field_type is type_ is t.Pattern:
            schema_.update(t_schema)
            break


def attrs_components_schema(attr_model: type[AttrsInstance]) -> dict[str, Schema]:
    flat_models = get_flat_attrs_from_attr_model(attr_model)
    attrs_name_map = get_attrs_name_map(flat_models)

    # gets model definitions
    definitions: dict[str, dict[str, t.Any]] = {}
    for model in flat_models:
        m_schema, m_definitions, _ = attrs_process_schema(
            model, attrs_name_map=attrs_name_map
        )
        definitions.update(m_definitions)
        model_name = attrs_name_map[model]
        definitions[model_name] = m_schema
    return {k: Schema(**definitions[k]) for k in sorted(definitions)}


# NOTE: Pydantic processing here.


def pydantic_components_schema(pydantic_model: type[pydantic.BaseModel]):
    if pkg_version_info("pydantic")[0] >= 2:
        return pydantic_v2_components_schema(pydantic_model)
    else:
        return pydantic_v1_components_schema(pydantic_model)


def pydantic_v2_components_schema(pydantic_model: type[pydantic.BaseModel]):
    # XXX: This is currently a hack until I learn more about pydantic's new
    # JSON validator, but it works for now.
    json_schema = jschema.model_json_schema(
        pydantic_model, ref_template=REF_PREFIX + "{model}"
    )
    defs = json_schema.pop("$defs", None)
    components: dict[str, Schema] = {pydantic_model.__name__: Schema(**json_schema)}
    if defs is not None:
        # NOTE: This is a nested models, hence we will update the definitions
        components.update({k: Schema(**v) for k, v in defs.items()})
    return components


def pydantic_v1_components_schema(pydantic_model: type[pydantic.BaseModel]):
    flat_models = schema.get_flat_models_from_model(pydantic_model)
    model_name_map = schema.get_model_name_map(flat_models)

    # gets model definitions
    definitions: dict[str, dict[str, t.Any]] = {}
    for model in flat_models:
        m_schema, m_definitions, _ = schema.model_process_schema(
            model, model_name_map=model_name_map, ref_prefix=REF_PREFIX
        )
        definitions.update(m_definitions)
        model_name = model_name_map[model]
        definitions[model_name] = m_schema
    return {k: Schema(**definitions[k]) for k in sorted(definitions)}


@lru_cache(maxsize=1)
def exception_components_schema() -> dict[str, Schema]:
    return {
        schema.title: schema
        for ex in [InvalidArgument, NotFound, InternalServerError]
        for schema in exception_schema(ex)
    }


def exception_schema(ex: t.Type[BentoMLException]) -> t.Iterable[FilledExceptionSchema]:
    # convert BentoML exception to OpenAPI components schema
    error_properties = {
        "msg": Schema(title="Message", type="string"),
        "type": Schema(title="Error Type", type="string"),
    }

    yield FilledExceptionSchema(
        title=ex.__name__,
        type="object",
        description=ex.error_code.phrase,
        properties=error_properties,
        required=list(error_properties),
    )


class FilledExceptionSchema(Schema):
    title: str
    description: str
