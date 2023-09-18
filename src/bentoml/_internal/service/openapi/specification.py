"""
Python representation of the OpenAPI Specification.

Specs can be found at https://spec.openapis.org/oas/v<semver>#openapi-specification (semver := 3.x.x | 2.x.x)

We will refer to the Python object that coresponds to an OpenAPI object as POS (Python implementation for a OpenAPI Specification).

Note that even though we cover most bases, there are still a lot of OpenAPI features such as deprecation,
webhooks, securities, etc. are yet to be implemented/exposed to user.
"""
from __future__ import annotations

import typing as t
import logging

import attr
import yaml
import cattr.errors
from cattr.gen import override
from cattr.gen import make_dict_unstructure_fn

from ...utils import bentoml_cattr

logger = logging.getLogger(__name__)

_T = t.TypeVar("_T")


@attr.frozen
class Contact:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    name: t.Optional[str] = None
    url: t.Optional[str] = None
    email: t.Optional[str] = None


@attr.frozen
class ExternalDocumentation:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    url: str
    description: t.Optional[str] = None


@attr.frozen
class Link:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    operationRef: t.Optional[str] = None
    operationId: t.Optional[str] = None
    requestBody: t.Optional[t.Any] = None
    description: t.Optional[str] = None

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
    mapping: t.Optional[t.Dict[str, str]] = None


@attr.frozen
class Schema:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False

    __rename_fields__ = {"ref": "$ref", "not_": "not"}

    type: t.Optional[str]
    ref: t.Optional[str] = None
    title: t.Optional[str] = None
    multipleOf: t.Optional[float] = None
    maximum: t.Optional[float] = None
    exclusiveMaximum: t.Optional[float] = None
    minimum: t.Optional[float] = None
    exclusiveMinimum: t.Optional[float] = None
    maxLength: t.Optional[int] = None
    minLength: t.Optional[int] = None
    pattern: t.Optional[str] = None
    maxItems: t.Optional[int] = None
    minItems: t.Optional[int] = None
    uniqueItems: t.Optional[bool] = None
    maxProperties: t.Optional[int] = None
    minProperties: t.Optional[int] = None
    required: t.Optional[t.List[str]] = None
    enum: t.Optional[t.List[t.Any]] = None
    allOf: t.Optional[t.List[Schema]] = None
    oneOf: t.Optional[t.List[Schema]] = None
    anyOf: t.Optional[t.List[Schema]] = None
    not_: t.Optional[Schema] = None
    items: t.Optional[t.Union[Schema, t.List[Schema]]] = None
    properties: t.Optional[t.Dict[str, t.Union[Schema, Reference]]] = None
    additionalProperties: t.Optional[t.Union[Schema, Reference, bool]] = None
    description: t.Optional[str] = None
    format: t.Optional[str] = None
    default: t.Optional[t.Any] = None
    nullable: t.Optional[bool] = None
    discriminator: t.Optional[Discriminator] = None
    readOnly: t.Optional[bool] = None
    writeOnly: t.Optional[bool] = None
    externalDocs: t.Optional[ExternalDocumentation] = None
    example: t.Optional[t.Any] = None
    deprecated: t.Optional[bool] = None
    # not yet supported: xml


@attr.frozen
class Example:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    summary: t.Optional[str] = None
    description: t.Optional[str] = None
    value: t.Optional[t.Any] = None
    externalValue: t.Optional[str] = None


@attr.frozen
class Encoding:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    contentType: t.Optional[str] = None
    style: t.Optional[str] = None
    explode: t.Optional[bool] = None
    allowReserved: t.Optional[bool] = None

    # not yet supported: headers


@attr.frozen
class MediaType:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    schema: t.Optional[t.Union[Schema, Reference]]
    example: t.Optional[t.Any] = None
    examples: t.Optional[t.Dict[str, t.Union[Example, Reference]]] = None
    encoding: t.Optional[t.Dict[str, Encoding]] = None


@attr.frozen
class Response:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    description: str
    content: t.Optional[t.Dict[str, MediaType]] = None
    links: t.Optional[t.Dict[str, t.Union[Link, Reference]]] = None

    # not yet supported: headers


@attr.frozen
class RequestBody:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    content: t.Dict[str, MediaType]
    description: t.Optional[str] = None
    required: t.Optional[bool] = None


@attr.frozen
class Operation:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    responses: t.Dict[t.Union[str, int], t.Union[Response, Reference]]
    tags: t.Optional[t.List[t.Union[str, Tag]]] = None
    summary: t.Optional[str] = None
    description: t.Optional[str] = None
    externalDocs: t.Optional[ExternalDocumentation] = None
    operationId: t.Optional[str] = None
    requestBody: t.Optional[t.Union[RequestBody, Reference, t.Dict[str, t.Any]]] = None

    # Not yet supported: parameters, callbacks, deprecated, servers, security


@attr.frozen
class Info:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    __preserve_cls_structure__ = True

    title: str
    version: str
    description: t.Optional[str] = None
    contact: t.Optional[Contact] = None

    # Not yet supported: termsOfService


@attr.frozen
class PathItem:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    __preserve_cls_structure__ = True

    __rename_fields__ = {"ref": "$ref"}

    ref: t.Optional[str] = None
    summary: t.Optional[str] = None
    description: t.Optional[str] = None
    get: t.Optional[t.Union[Operation, t.Dict[str, t.Any]]] = None
    put: t.Optional[t.Union[Operation, t.Dict[str, t.Any]]] = None
    post: t.Optional[t.Union[Operation, t.Dict[str, t.Any]]] = None
    delete: t.Optional[t.Union[Operation, t.Dict[str, t.Any]]] = None
    options: t.Optional[t.Union[Operation, t.Dict[str, t.Any]]] = None
    head: t.Optional[t.Union[Operation, t.Dict[str, t.Any]]] = None
    patch: t.Optional[t.Union[Operation, t.Dict[str, t.Any]]] = None
    trace: t.Optional[t.Union[Operation, t.Dict[str, t.Any]]] = None
    # not yet supported: servers, parameters


@attr.frozen
class Tag:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    __preserve_cls_structure__ = True

    name: str
    description: t.Optional[str] = None
    externalDocs: t.Optional[ExternalDocumentation] = None


@attr.frozen
class Components:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    __preserve_cls_structure__ = True

    schemas: t.Dict[str, t.Union[Schema, Reference]]
    responses: t.Optional[t.Dict[str, t.Union[Response, Reference]]] = None
    examples: t.Optional[t.Dict[str, t.Union[Example, Reference]]] = None
    requestBodies: t.Optional[
        t.Dict[str, t.Union[RequestBody, Reference, t.Dict[str, t.Any]]]
    ] = None
    links: t.Optional[t.Dict[str, t.Union[Link, Reference]]] = None

    # Not yet supported: securitySchemes, callbacks, parameters, headers

    def asdict(self) -> t.Dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)


@attr.frozen
class OpenAPISpecification:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    openapi: str
    info: Info
    paths: t.Dict[str, PathItem]
    servers: t.List[t.Any]
    tags: t.Optional[t.List[Tag]] = None
    components: t.Optional[Components] = None

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


def _structure_rename_fields_hook(data: t.Dict[str, t.Any], cl: t.Type[_T]) -> _T:
    # pop is atomic, so we don't need to worry about performance deficit.
    # See https://stackoverflow.com/a/17326099/8643197.
    rev = {
        k: data.pop(v) for k, v in getattr(cl, "__rename_fields__").items() if v in data
    }
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
def _preserve_cls_structure(data: dict[str, t.Any], cl: t.Type[_T]) -> _T:
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
