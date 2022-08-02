"""
Python representation of the OpenAPI Specification.

Specs can be found at https://spec.openapis.org/oas/v<semver>#openapi-specification (semver := 3.x.x | 2.x.x)

We will refers to the Python object that coresponds to an OpenAPI object as POS (Python implementation for a OpenAPI Specification).

Note that even though we cover most bases, there are still a lot of OpenAPI features such as deprecation,
webhooks, securities, etc. are yet to be implemented/exposed to user.
"""
from __future__ import annotations

import enum
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

# All classes contains the following special fields:
# - __omit_if_default__: If True, cattrs will omit all fields that use default values. This is useful for most POS
#                        as the default value for most fields are set to None.
# - __forbid_extra_keys__: If True, cattrs will raise an error if given data/dictionary contains additional keys that is not an attribute of a given POS.
# - __rename_fields__: a mapping of existing fields and the correct name specified under OpenAPI specification.
#                      To avoid naming collision with Python builtin types as well as reserved words, we will postfix
#                      the field name with an underscore. __rename_fields__'s responsibility is to map the correct name per OpenAPI specification.
# - __preserve_cls_structure__: If True, cattrs will allow to return the OpenAPI object directly.


class SecuritySchemeType(enum.Enum):
    apiKey = "apiKey"
    http = "http"
    oauth2 = "oauth2"
    openIdConnect = "openIdConnect"


class APIKeyIn(enum.Enum):
    query = "query"
    header = "header"
    cookie = "cookie"


@attr.frozen
class APIKey:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    __rename_fields__ = {"in_": "in"}

    name: str
    in_: t.Optional[APIKeyIn]
    description: t.Optional[str] = None
    type: t.Literal[SecuritySchemeType.apiKey] = SecuritySchemeType.apiKey


@attr.frozen
class HTTPBase:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    scheme: str
    description: t.Optional[str] = None
    type: t.Literal[SecuritySchemeType.http] = SecuritySchemeType.http


@attr.frozen
class HTTPBearer(HTTPBase):
    scheme: str = "bearer"
    bearerFormat: t.Optional[str] = None


@attr.frozen
class OAuthFlowImplicit:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    authorizationUrl: str
    refreshUrl: t.Optional[str] = None
    scopes: t.Dict[str, str] = attr.field(factory=dict)


@attr.frozen
class OAuthFlowPassword:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    tokenUrl: str
    refreshUrl: t.Optional[str] = None
    scopes: t.Dict[str, str] = attr.field(factory=dict)


@attr.frozen
class OAuthFlowClientCredentials:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    tokenUrl: str
    refreshUrl: t.Optional[str] = None
    scopes: t.Dict[str, str] = attr.field(factory=dict)


@attr.frozen
class OAuthFlowAuthorizationCode:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    authorizationUrl: str
    tokenUrl: str
    refreshUrl: t.Optional[str] = None
    scopes: t.Dict[str, str] = attr.field(factory=dict)


@attr.frozen
class OAuthFlows:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    implicit: t.Optional[OAuthFlowImplicit] = None
    password: t.Optional[OAuthFlowPassword] = None
    clientCredentials: t.Optional[OAuthFlowClientCredentials] = None
    authorizationCode: t.Optional[OAuthFlowAuthorizationCode] = None


@attr.frozen
class OAuth2:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    flows: OAuthFlows
    description: t.Optional[str] = None
    type: t.Literal[SecuritySchemeType.oauth2] = SecuritySchemeType.oauth2


@attr.frozen
class OpenIdConnect:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    openIdConnectUrl: str
    description: t.Optional[str] = None
    type: t.Literal[SecuritySchemeType.oauth2] = SecuritySchemeType.oauth2


SecurityScheme = t.Union[APIKey, HTTPBase, OAuth2, OpenIdConnect, HTTPBearer]


@attr.frozen
class Contact:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    name: t.Optional[str] = None
    url: t.Optional[str] = None
    email: t.Optional[str] = None


@attr.frozen
class License:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    name: str
    identifier: t.Optional[str] = attr.field(
        default=attr.Factory(lambda self: self.name.replace(" ", "-"), takes_self=True),
    )
    url: t.Optional[str] = None


Apache2_0 = License(
    name="Apache 2.0",
    url="https://www.apache.org/licenses/LICENSE-2.0.html",
)


@attr.frozen
class ExternalDocumentation:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    url: str
    description: t.Optional[str] = None


@attr.frozen
class ServerVariable:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    default: str
    enum: t.Optional[t.List[str]] = None
    description: t.Optional[str] = None


@attr.frozen
class Server:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    url: str
    description: t.Optional[str] = None
    variables: t.Optional[t.Dict[str, ServerVariable]] = None


@attr.frozen
class Link:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    operationRef: t.Optional[str] = None
    operationId: t.Optional[str] = None
    parameters: t.Optional[t.Dict[str, t.Any]] = None
    requestBody: t.Optional[t.Any] = None
    description: t.Optional[str] = None
    server: t.Optional[Server] = None


class ParameterInType(enum.Enum):
    query = "query"
    header = "header"
    path = "path"
    cookie = "cookie"


@attr.frozen
class ParamBase:
    __omit_if_default__ = True
    __forbid_extra_keys__ = False
    __rename_fields__ = {"in_": "in"}

    name: t.Optional[str]
    in_: t.Optional[ParameterInType] = None
    description: t.Optional[str] = None
    required: t.Optional[bool] = None
    deprecated: t.Optional[bool] = None
    style: t.Optional[str] = None
    explode: t.Optional[bool] = None
    allowReserved: t.Optional[bool] = None
    schema: t.Optional[t.Union[Schema, Reference]] = None
    example: t.Optional[t.Any] = None
    examples: t.Optional[t.Dict[str, t.Union[Example, Reference]]] = None
    content: t.Optional[t.Dict[str, MediaType]] = None


class Parameter(ParamBase):
    pass


class Header(ParamBase):
    name = None
    in_ = None


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
class XML:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    name: t.Optional[str] = None
    namespace: t.Optional[str] = None
    prefix: t.Optional[str] = None
    attribute: t.Optional[bool] = None
    wrapped: t.Optional[bool] = None


@attr.frozen
class Schema:
    """
    The Schema Object allows the definition of input and output data types.
    These types can be objects, but also primitives and arrays. This object is an extended subset of http://json-schema.org/.

    ground truth: https://spec.openapis.org/oas/v3.0.2#schema-object
    """

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
    properties: t.Optional[t.Dict[str, Schema]] = None
    additionalProperties: t.Optional[t.Union[Schema, Reference, bool]] = None
    description: t.Optional[str] = None
    format: t.Optional[str] = None
    default: t.Optional[t.Any] = None
    nullable: t.Optional[bool] = None
    discriminator: t.Optional[Discriminator] = None
    readOnly: t.Optional[bool] = None
    writeOnly: t.Optional[bool] = None
    xml: t.Optional[XML] = None
    externalDocs: t.Optional[ExternalDocumentation] = None
    example: t.Optional[t.Any] = None
    deprecated: t.Optional[bool] = None


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
    headers: t.Optional[t.Dict[str, t.Union[Header, Reference]]] = None
    style: t.Optional[str] = None
    explode: t.Optional[bool] = None
    allowReserved: t.Optional[bool] = None


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
    headers: t.Optional[t.Dict[str, t.Union[Header, Reference]]] = None
    # TODO: validate str to correct media type
    content: t.Optional[t.Dict[str, MediaType]] = None
    links: t.Optional[t.Dict[str, t.Union[Link, Reference]]] = None


@attr.frozen
class RequestBody:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    content: t.Dict[str, MediaType]
    description: t.Optional[str] = None
    required: t.Optional[bool] = None


# represents a "get", "post", "put", "delete", "options", "head", "patch", "trace" operations.
@attr.frozen
class Operation:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    responses: t.Dict[str, t.Union[Response, Reference]]
    tags: t.Optional[t.List[str | Tag]] = None
    summary: t.Optional[str] = None
    description: t.Optional[str] = None
    externalDocs: t.Optional[ExternalDocumentation] = None
    operationId: t.Optional[str] = None
    parameters: t.Optional[t.List[t.Union[Parameter, Reference]]] = None
    requestBody: t.Optional[t.Union[RequestBody, Reference]] = None

    # Note that this is not yet implemented.
    callbacks: t.Optional[
        t.Mapping[str, t.Union[t.Dict[str, PathItem], Reference]]
    ] = None
    deprecated: t.Optional[bool] = None
    security: t.Optional[t.Dict[str, t.List[str]]] = None
    servers: t.Optional[t.List[Server]] = None


@attr.frozen
class Info:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    title: str
    version: str
    summary: t.Optional[str] = None
    description: t.Optional[str] = None
    termsOfService: t.Optional[str] = None
    contact: t.Optional[Contact] = None
    license: t.Optional[License] = None

    __preserve_cls_structure__ = True


@attr.frozen
class PathItem:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    __rename_fields__ = {"ref": "$ref"}

    ref: t.Optional[str] = None
    summary: t.Optional[str] = None
    description: t.Optional[str] = None
    get: t.Optional[Operation] = None
    put: t.Optional[Operation] = None
    post: t.Optional[Operation] = None
    delete: t.Optional[Operation] = None
    options: t.Optional[Operation] = None
    head: t.Optional[Operation] = None
    patch: t.Optional[Operation] = None
    trace: t.Optional[Operation] = None
    servers: t.Optional[t.List[Server]] = None
    parameters: t.Optional[t.List[t.Union[Parameter, Reference]]] = None

    __preserve_cls_structure__ = True


@attr.frozen
class Tag:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    name: str
    description: t.Optional[str] = None
    externalDocs: t.Optional[ExternalDocumentation] = None

    __preserve_cls_structure__ = True


@attr.frozen
class Components:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    schemas: t.Dict[str, t.Union[Schema, Reference]]

    # The below fields are not yet implemented.
    # Here for consistency sake.
    responses: t.Optional[t.Dict[str, t.Union[Response, Reference]]] = None
    parameters: t.Optional[t.Dict[str, t.Union[Parameter, Reference]]] = None
    examples: t.Optional[t.Dict[str, t.Union[Example, Reference]]] = None
    requestBodies: t.Optional[t.Dict[str, t.Union[RequestBody, Reference]]] = None
    headers: t.Optional[t.Dict[str, t.Union[Header, Reference]]] = None
    securitySchemes: t.Optional[t.Dict[str, t.Union[SecurityScheme, Reference]]] = None
    links: t.Optional[t.Dict[str, t.Union[Link, Reference]]] = None
    # Using Any for Specification Extensions
    callbacks: t.Optional[
        t.Dict[str, t.Union[t.Dict[str, PathItem], Reference, t.Any]]
    ] = None


@attr.frozen
class OpenAPISpecification:
    __omit_if_default__ = True
    __forbid_extra_keys__ = True

    openapi: str
    info: Info
    paths: t.Dict[str, PathItem]
    tags: t.Optional[t.List[Tag]] = None
    components: t.Optional[Components] = None

    # EXPERIMENTAL/NOT-YET-IMPLEMENTED
    servers: t.Optional[t.List[Server]] = None
    security: t.Optional[t.List[t.Dict[str, t.List[str]]]] = None
    externalDocs: t.Optional[ExternalDocumentation] = None
    jsonSchemaDialect: t.Optional[str] = None
    webhooks: t.Any = None

    def to_dict(self) -> dict[str, t.Any]:
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
    return dumper.represent_dict(spec.to_dict())


yaml.add_representer(OpenAPISpecification, _OpenAPISpecification_dumper)
