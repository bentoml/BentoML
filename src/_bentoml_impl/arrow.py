from __future__ import annotations

import typing as t

import pyarrow as pa
from _bentoml_sdk.types import arrow_serialization
from pydantic import BaseModel

SchemaDict: t.TypeAlias = t.Dict[str, t.Any]

T = t.TypeVar("T", bound=BaseModel)


def model_to_arrow_schema(model: type[BaseModel]) -> pa.Schema:
    schema = model.model_json_schema(mode="serialization")
    fields = _model_to_fields(schema, ref_defs=schema.get("$defs", {}))
    return pa.schema(fields)


def _model_to_fields(
    model_schema: SchemaDict, ref_defs: dict[str, SchemaDict]
) -> list[pa.Field]:
    return [
        _field_to_arrow(name, field, ref_defs)
        for name, field in model_schema.get("properties", {}).items()
    ]


def _type_to_arrow(field_type: str | None) -> pa.DataType:
    if field_type == "integer":
        return pa.int64()
    elif field_type == "number":
        return pa.float64()
    elif field_type == "string":
        return pa.utf8()
    elif field_type == "boolean":
        return pa.bool_()
    raise TypeError(
        f"Converting Pydantic type to Arrow Type: unsupported type {field_type}"
    )


REFS_PREFIX = "#/$defs/"


def _field_schema_to_arrow(
    field: SchemaDict, ref_defs: dict[str, SchemaDict]
) -> pa.DataType:
    if "$ref" in field:
        ref_name = field["$ref"][len(REFS_PREFIX) :]
        field = ref_defs[ref_name]
    field_type = field.get("type")
    if is_nullable(field):
        return _field_schema_to_arrow(field["anyOf"][0], ref_defs)
    elif field_type == "array":
        return pa.list_(_field_schema_to_arrow(field["items"], ref_defs))
    elif field_type == "object" and "properties" in field:
        return pa.struct(_model_to_fields(field, ref_defs))
    elif field_type == "object" and "additionalProperties" in field:
        return pa.map_(
            pa.utf8(), _field_schema_to_arrow(field["additionalProperties"], ref_defs)
        )
    elif field_type == "tensor":
        dtype = field.get("dtype")
        if dtype is None:
            raise ValueError("dtype must be specified")
        child_type = getattr(pa, dtype)()
        return pa.list_(child_type)
    elif field_type == "string" and field.get("format") == "date-time":
        return pa.timestamp("ms")
    elif field_type == "string" and field.get("format") == "binary":
        return pa.binary()
    return _type_to_arrow(field_type)


def is_nullable(field: SchemaDict) -> bool:
    """Check if a Pydantic FieldInfo is nullable."""
    return (
        "anyOf" in field
        and len(field["anyOf"]) == 2
        and {"type": "null"} in field["anyOf"]
    )


def _field_to_arrow(
    name: str, field: SchemaDict, ref_defs: dict[str, SchemaDict]
) -> pa.Field:
    return pa.field(name, _field_schema_to_arrow(field, ref_defs), is_nullable(field))


def serialize_to_arrow(model: BaseModel, out_stream: t.BinaryIO) -> None:
    arrow_schema = model_to_arrow_schema(model.__class__)
    with arrow_serialization():
        table = pa.Table.from_pylist([model.model_dump()], schema=arrow_schema)
    with pa.ipc.new_stream(out_stream, arrow_schema) as writer:
        writer.write_table(table)


def deserialize_from_arrow(model: type[T], in_stream: t.BinaryIO) -> T:
    with pa.ipc.open_stream(in_stream) as reader:
        df = reader.read_pandas()
        ins = df.to_dict(orient="records")[0]
    return model(**ins)
