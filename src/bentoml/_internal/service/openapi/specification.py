"""
Python representation of the OpenAPI Specification.

Specs can be found at https://spec.openapis.org/oas/v<semver>#openapi-specification (semver := 3.x.x | 2.x.x)

We will refer to the Python object that coresponds to an OpenAPI object as POS (Python implementation for a OpenAPI Specification).

Note that even though we cover most bases, there are still a lot of OpenAPI features such as deprecation,
webhooks, securities, etc. are yet to be implemented/exposed to user.
"""

from __future__ import annotations

import logging
import typing as t

import attr
import cattr.errors
import yaml
from cattr.gen import make_dict_unstructure_fn
from cattr.gen import override

from ...utils.cattr import bentoml_cattr

logger = logging.getLogger(__name__)

_T = t.TypeVar("_T")


@attr.frozen
class Contact:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    name: str | None = None
    url: str | None = None
    email: str | None = None


@attr.frozen
class ExternalDocumentation:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    url: str
    description: str | None = None


@attr.frozen
class Link:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    operationRef: str | None = None
    operationId: str | None = None
    requestBody: t.Any | None = None
    description: str | None = None

    # not yet supported: parameters


@attr.frozen
class Reference:
    __rename_fields__ = {"ref": "$ref"}

    ref: str


@attr.frozen
class Discriminator:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    propertyName: str
    mapping: dict[str, str] | None = None


@attr.frozen
class Schema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False

    __rename_fields__ = {"ref": "$ref", "not_": "not"}

    type: str | None = None
    ref: str | None = None
    title: str | None = None
    multipleOf: float | None = None
    maximum: float | None = None
    exclusiveMaximum: float | None = None
    minimum: float | None = None
    exclusiveMinimum: float | None = None
    maxLength: int | None = None
    minLength: int | None = None
    pattern: str | None = None
    maxItems: int | None = None
    minItems: int | None = None
    uniqueItems: bool | None = None
    prefixItems: list[Schema] | None = None
    maxProperties: int | None = None
    minProperties: int | None = None
    required: list[str] | None = None
    enum: list[t.Any] | None = None
    allOf: list[Schema] | None = None
    oneOf: list[Schema] | None = None
    anyOf: list[Schema] | None = None
    not_: Schema | None = None
    items: Schema | list[Schema] | None = None
    properties: dict[str, Schema | Reference] | None = None
    additionalProperties: Schema | Reference | bool | None = None
    description: str | None = None
    format: str | None = None
    default: t.Any | None = None
    nullable: bool | None = None
    discriminator: Discriminator | None = None
    readOnly: bool | None = None
    writeOnly: bool | None = None
    externalDocs: ExternalDocumentation | None = None
    example: t.Any | None = None
    deprecated: bool | None = None
    root_input: bool = False
    # not yet supported: xml


@attr.frozen
class Example:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    summary: str | None = None
    description: str | None = None
    value: t.Any | None = None
    externalValue: str | None = None


@attr.frozen
class Encoding:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    contentType: str | None = None
    style: str | None = None
    explode: bool | None = None
    allowReserved: bool | None = None

    # not yet supported: headers


@attr.frozen
class MediaType:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    schema: Schema | Reference | None
    example: t.Any | None = None
    examples: dict[str, Example | Reference] | None = None
    encoding: dict[str, Encoding] | None = None


@attr.frozen
class Response:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    description: str
    content: dict[str, MediaType] | None = None
    links: dict[str, Link | Reference] | None = None

    # not yet supported: headers


@attr.frozen
class RequestBody:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    content: dict[str, MediaType]
    description: str | None = None
    required: bool | None = None


@attr.frozen
class Operation:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    responses: dict[str | int, Response | Reference]
    tags: list[str | Tag] | None = None
    summary: str | None = None
    description: str | None = None
    externalDocs: ExternalDocumentation | None = None
    operationId: str | None = None
    requestBody: RequestBody | Reference | dict[str, t.Any] | None = None

    # Not yet supported: parameters, callbacks, deprecated, servers, security


@attr.frozen
class Info:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    __preserve_cls_structure__ = True

    title: str
    version: str
    description: str | None = None
    contact: Contact | None = None

    # Not yet supported: termsOfService


@attr.frozen
class PathItem:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    __preserve_cls_structure__ = True

    __rename_fields__ = {"ref": "$ref"}

    ref: str | None = None
    summary: str | None = None
    description: str | None = None
    get: Operation | dict[str, t.Any] | None = None
    put: Operation | dict[str, t.Any] | None = None
    post: Operation | dict[str, t.Any] | None = None
    delete: Operation | dict[str, t.Any] | None = None
    options: Operation | dict[str, t.Any] | None = None
    head: Operation | dict[str, t.Any] | None = None
    patch: Operation | dict[str, t.Any] | None = None
    trace: Operation | dict[str, t.Any] | None = None
    # not yet supported: servers, parameters


@attr.frozen
class Tag:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    __preserve_cls_structure__ = True

    name: str
    description: str | None = None
    externalDocs: ExternalDocumentation | None = None


@attr.frozen
class Components:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    __preserve_cls_structure__ = True

    schemas: dict[str, Schema | Reference]
    responses: dict[str, Response | Reference] | None = None
    examples: dict[str, Example | Reference] | None = None
    requestBodies: dict[str, RequestBody | Reference | dict[str, t.Any]] | None = None
    links: dict[str, Link | Reference] | None = None

    # Not yet supported: securitySchemes, callbacks, parameters, headers

    def asdict(self) -> dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)


@attr.frozen
class OpenAPISpecification:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    openapi: str
    info: Info
    paths: dict[str, PathItem]
    servers: list[t.Any]
    tags: list[Tag] | None = None
    components: Components | None = None

    # Not yet supported: servers, security, externalDocs, webhooks, jsonSchemaDialect

    def asdict(self) -> dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)

    @classmethod
    def from_yaml_file(cls, stream: t.IO[t.Any]) -> OpenAPISpecification:
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise

        try:
            return bentoml_cattr.structure(yaml_content, cls)
        except cattr.errors.ClassValidationError:
            raise


def _structure_rename_fields_hook(data: dict[str, t.Any], cl: type[_T]) -> _T:
    # pop is atomic, so we don't need to worry about performance deficit.
    # See https://stackoverflow.com/a/17326099/8643197.
    rev = {k: data.pop(v) for k, v in cl.__rename_fields__.items() if v in data}
    return cl(**rev, **data)


# handles all OpenAPI class that includes __rename_fields__
bentoml_cattr.register_structure_hook_func(
    lambda cls: attr.has(cls) and hasattr(cls, "__rename_fields__"),
    lambda data, cl: _structure_rename_fields_hook(data, cl),
)
bentoml_cattr.register_unstructure_hook_factory(
    lambda cls: attr.has(cls) and hasattr(cls, "__rename_fields__"),
    lambda cls: make_dict_unstructure_fn(
        cls,
        bentoml_cattr,
        # for all classes under OpenAPI, we want to omit default values.
        _cattrs_omit_if_default=getattr(cls, "__omit_if_default__", True),
        **{k: override(rename=v) for k, v in cls.__rename_fields__.items()},
    ),
)


# register all class in this structure whom
# implement a '__preserve_cls_structure__' method
def _preserve_cls_structure(data: dict[str, t.Any], cl: type[_T]) -> _T:
    if isinstance(data, cl):
        return data
    return cl(**data)


bentoml_cattr.register_structure_hook_func(
    lambda cls: attr.has(cls) and hasattr(cls, "__preserve_cls_structure__"),
    lambda data, cls: _preserve_cls_structure(data, cls),
)


def _OpenAPISpecification_dumper(
    dumper: yaml.Dumper, spec: OpenAPISpecification
) -> yaml.Node:
    return dumper.represent_dict(spec.asdict())


yaml.add_representer(OpenAPISpecification, _OpenAPISpecification_dumper)
