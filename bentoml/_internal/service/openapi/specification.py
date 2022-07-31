from __future__ import annotations

import enum
import typing as t

import attr

from bentoml._internal.utils import bentoml_cattr


@attr.frozen
class Info:
    __forbid_extra_keys__ = True

    title: str
    summary: str
    version: str
    description: t.Optional[str] = None
    termsOfService: t.Optional[str] = None
    contact: t.Optional[Contact] = None
    license: t.Optional[License] = None


@attr.frozen
class Contact:
    name: t.Optional[str] = None
    url: t.Optional[str] = None
    email: t.Optional[str] = None


@attr.frozen
class License:
    name: str
    identifier: t.Optional[str] = attr.field(
        default=attr.Factory(lambda self: self.name.replace(" ", "-"), takes_self=True),
    )
    url: t.Optional[str] = None


@attr.frozen
class ExternalDocumentation:
    url: str
    description: t.Optional[str] = None


@attr.frozen
class Tag:
    name: str
    description: t.Optional[str] = None
    externalDocs: t.Optional[ExternalDocumentation] = None


@attr.frozen
class ServerVariable:
    default: str
    enum: t.Optional[t.List[str]] = None
    description: t.Optional[str] = None


@attr.frozen
class Server:
    url: str
    description: t.Optional[str] = None
    variables: t.Optional[t.Dict[str, ServerVariable]] = None


@attr.frozen
class Link:
    operationRef: t.Optional[str] = None
    operationId: t.Optional[str] = None
    parameters: t.Optional[t.Dict[str, t.Any]] = None
    requestBody: t.Optional[t.Any] = None
    description: t.Optional[str] = None
    server: t.Optional[Server] = None


@attr.frozen
class ParamBase:
    name: t.Optional[str]
    in_: t.Optional[ParameterInType]
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


class ParameterInType(enum.Enum):
    query = "query"
    header = "header"
    path = "path"
    cookie = "cookie"


class Parameter(ParamBase):
    pass


class Header(ParamBase):
    name = None
    in_ = None


@attr.frozen
class Reference:
    ref: str


def reference_structure_hook(
    data: t.Dict[str, t.Any], cl: t.Type[Reference]
) -> Reference:
    return cl(ref=data["$ref"])


def reference_unstructure_hook(o: Reference) -> dict[str, str]:
    return {"$ref": o.ref}


bentoml_cattr.register_structure_hook(Reference, reference_structure_hook)
bentoml_cattr.register_unstructure_hook(Reference, reference_unstructure_hook)


@attr.frozen
class Discriminator:
    propertyName: str
    mapping: t.Optional[t.Dict[str, str]] = None


@attr.frozen
class XML:
    name: t.Optional[str] = None
    namespace: t.Optional[str] = None
    prefix: t.Optional[str] = None
    attribute: t.Optional[bool] = None
    wrapped: t.Optional[bool] = None


@attr.frozen
class Schema:
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
    type: t.Optional[t.List[str]] = None
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
    summary: t.Optional[str] = None
    description: t.Optional[str] = None
    value: t.Optional[t.Any] = None
    externalValue: t.Optional[str] = None


@attr.frozen
class Encoding:
    contentType: t.Optional[str] = None
    headers: t.Optional[t.Dict[str, t.Union[Header, Reference]]] = None
    style: t.Optional[str] = None
    explode: t.Optional[bool] = None
    allowReserved: t.Optional[bool] = None


@attr.frozen
class MediaType:
    schema: t.Optional[t.Union[Schema, Reference]] = None
    example: t.Optional[t.Any] = None
    examples: t.Optional[t.Dict[str, t.Union[Example, Reference]]] = None
    encoding: t.Optional[t.Dict[str, Encoding]] = None


@attr.frozen
class Response:
    description: str
    headers: t.Optional[t.Dict[str, t.Union[Header, Reference]]] = None
    content: t.Optional[t.Dict[str, MediaType]] = None
    links: t.Optional[t.Dict[str, t.Union[Link, Reference]]] = None


@attr.frozen
class RequestBody:
    content: t.Dict[str, MediaType]
    description: t.Optional[str] = None
    required: t.Optional[bool] = None


@attr.frozen
class Operation:
    responses: t.Dict[str, t.Union[Response, Reference]]
    tags: t.Optional[t.List[str]] = None
    summary: t.Optional[str] = None
    description: t.Optional[str] = None
    externalDocs: t.Optional[ExternalDocumentation] = None
    operationId: t.Optional[str] = None
    parameters: t.Optional[t.List[t.Union[Parameter, Reference]]] = None
    requestBody: t.Optional[t.Union[RequestBody, Reference]] = None
    callbacks: t.Optional[
        t.Mapping[str, t.Union[t.Dict[str, PathItem], Reference]]
    ] = None
    deprecated: t.Optional[bool] = None
    security: t.Optional[t.Dict[str, t.List[str]]] = None
    servers: t.Optional[t.List[Server]] = None


@attr.frozen
class PathItem:
    __forbid_extra_keys__ = True

    ref: t.Optional[str] = None  # $ref
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


class SecuritySchemeType(enum.Enum):
    apiKey = "apiKey"
    http = "http"
    oauth2 = "oauth2"
    openIdConnect = "openIdConnect"


@attr.frozen
class APIKey:
    name: str
    in_: APIKeyIn
    description: t.Optional[str] = None
    type: t.Literal[SecuritySchemeType.apiKey] = SecuritySchemeType.apiKey


@attr.frozen
class HTTPBase:
    scheme: str
    description: t.Optional[str] = None
    type: t.Literal[SecuritySchemeType.http] = SecuritySchemeType.http


@attr.frozen
class HTTPBearer(HTTPBase):
    scheme: str = "bearer"
    bearerFormat: t.Optional[str] = None


@attr.frozen
class OAuthFlowImplicit:
    authorizationUrl: str
    refreshUrl: t.Optional[str] = None
    scopes: t.Dict[str, str] = attr.field(factory=dict)


@attr.frozen
class OAuthFlowPassword:
    tokenUrl: str
    refreshUrl: t.Optional[str] = None
    scopes: t.Dict[str, str] = attr.field(factory=dict)


@attr.frozen
class OAuthFlowClientCredentials:
    tokenUrl: str
    refreshUrl: t.Optional[str] = None
    scopes: t.Dict[str, str] = attr.field(factory=dict)


@attr.frozen
class OAuthFlowAuthorizationCode:
    authorizationUrl: str
    tokenUrl: str
    refreshUrl: t.Optional[str] = None
    scopes: t.Dict[str, str] = attr.field(factory=dict)


@attr.frozen
class OAuthFlows:
    implicit: t.Optional[OAuthFlowImplicit] = None
    password: t.Optional[OAuthFlowPassword] = None
    clientCredentials: t.Optional[OAuthFlowClientCredentials] = None
    authorizationCode: t.Optional[OAuthFlowAuthorizationCode] = None


@attr.frozen
class OAuth2:
    flows: OAuthFlows
    description: t.Optional[str] = None
    type: t.Literal[SecuritySchemeType.oauth2] = SecuritySchemeType.oauth2


@attr.frozen
class OpenIdConnect:
    openIdConnectUrl: str
    description: t.Optional[str] = None
    type: t.Literal[SecuritySchemeType.oauth2] = SecuritySchemeType.oauth2


class APIKeyIn(enum.Enum):
    query = "query"
    header = "header"
    cookie = "cookie"


SecurityScheme = t.Union[APIKey, HTTPBase, OAuth2, OpenIdConnect, HTTPBearer]


@attr.frozen
class Components:
    __forbid_extra_keys__ = True

    schemas: t.Optional[t.Dict[str, t.Union[Schema, Reference]]] = None
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
    __forbid_extra_keys__ = True

    openapi: str
    info: Info
    paths: t.Dict[str, PathItem] = attr.field(
        validator=attr.validators.deep_mapping(
            key_validator=lambda _, __, value: value.startswith("/"),
            value_validator=lambda _, __, value: isinstance(value, dict),
        )
    )
    servers: t.Optional[t.List[Server]] = None
    components: t.Optional[Components] = None
    security: t.Optional[t.List[t.Dict[str, t.List[str]]]] = None
    tags: t.Optional[t.List[Tag]] = None
    externalDocs: t.Optional[ExternalDocumentation] = None

    # fields that are not yet implemented.
    # json_schema_dialect: str
    # webhooks: t.Any = None

    def asdict(self) -> dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)

    @classmethod
    def from_dict(cls, dct: dict[str, t.Any]) -> OpenAPISpecification:
        return bentoml_cattr.structure(dct, OpenAPISpecification)


if __name__ == "__main__":
    val = {
        "title": "Sample BentoService",
        "summary": "A sample BentoService",
        "description": "This is a sample BentoServer.",
        "termsOfService": "https://example.com/terms/",
        "contact": {
            "name": "API Support",
            "url": "https://www.example.com/support",
            "email": "support@example.com",
        },
        "license": {
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        "version": "1.0.1",
    }
    print(bentoml_cattr.structure(val, Info))
