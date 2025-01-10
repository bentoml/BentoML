from __future__ import annotations

import logging
import typing as t
from http import HTTPStatus
from typing import Any
from typing import Dict
from typing import TypeVar
from typing import Union
from typing import cast

from deepmerge.merger import Merger
from fastapi import FastAPI  # type: ignore
from fastapi.openapi.utils import get_openapi  # type: ignore
from pydantic import BaseModel as PydanticBaseModel  # type: ignore
from typing_extensions import TypedDict

from bentoml._internal.service.openapi import APP_TAG
from bentoml._internal.service.openapi import INFRA_TAG
from bentoml._internal.service.openapi import make_infra_endpoints
from bentoml._internal.service.openapi.specification import Components
from bentoml._internal.service.openapi.specification import Contact
from bentoml._internal.service.openapi.specification import Info
from bentoml._internal.service.openapi.specification import MediaType
from bentoml._internal.service.openapi.specification import OpenAPISpecification
from bentoml._internal.service.openapi.specification import PathItem
from bentoml._internal.service.openapi.specification import Reference
from bentoml._internal.service.openapi.specification import Response
from bentoml._internal.service.openapi.specification import Schema
from bentoml._internal.service.openapi.utils import exception_components_schema
from bentoml._internal.service.openapi.utils import exception_schema
from bentoml._internal.types import LazyType
from bentoml._internal.utils import bentoml_cattr
from bentoml.exceptions import InternalServerError
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import NotFound

logger = logging.getLogger(__name__)

# Type alias for better type checking
BaseModel = PydanticBaseModel  # type: ignore

T = TypeVar("T")


# Type aliases for better readability and type safety
class OpenAPIDict(TypedDict, total=False):
    paths: Dict[str, Any]
    components: Dict[str, Any]
    title: str
    version: str
    routes: list[Any]


class SchemaDict(TypedDict, total=False):
    type: str
    ref: str
    properties: Dict[str, Any]
    items: Dict[str, Any]


ComponentsDict = Dict[str, Dict[str, Union[Schema, Reference]]]
SchemaType = Union[Schema, Reference, Dict[str, Any]]

if t.TYPE_CHECKING:
    import fastapi as fastapi

    from .factory import Service

merger = Merger(
    # merge dicts recursively
    [(dict, "merge")],
    # override all other types (including lists)
    ["override"],
    # override conflicting types
    ["override"],
)

REF_TEMPLATE = "#/components/schemas/{model}"


def generate_spec(svc: Service[t.Any], *, openapi_version: str = "3.0.2"):
    """Generate a OpenAPI specification for a service."""
    mounted_app_paths: Dict[str, PathItem] = {}
    schema_components: Dict[str, Dict[str, SchemaType]] = {}

    def join_path(prefix: str, path: str) -> str:
        return f"{prefix.rstrip('/')}/{path.lstrip('/')}"

    for app, path, _ in svc.mount_apps:
        if LazyType["fastapi.FastAPI"]("fastapi.FastAPI").isinstance(app):
            app_instance = t.cast(FastAPI, app)

            openapi_spec: OpenAPIDict = get_openapi(
                title=app_instance.title,
                version=app_instance.version,
                routes=app_instance.routes,
            )
            mounted_app_paths.update(
                {
                    join_path(path, str(k)): bentoml_cattr.structure(v, PathItem)
                    for k, v in cast(
                        Dict[str, Any], openapi_spec.get("paths", {})
                    ).items()
                }
            )

            components = openapi_spec.get("components", {})
            if components:
                merger.merge(
                    schema_components,
                    cast(Dict[str, Dict[str, SchemaType]], components),
                )

    merger.merge(schema_components, generate_service_components(svc))

    # Convert schema components to proper type
    schemas: dict[str, t.Union[Schema, Reference]] = {}
    schema_dict = schema_components.get("schemas", {})
    for name, schema in schema_dict.items():
        if isinstance(schema, (Schema, Reference)):
            schemas[name] = schema
        elif isinstance(schema, dict):
            try:
                if "$ref" in schema:
                    schemas[name] = Reference(ref=schema["$ref"])
                else:
                    # Convert schema with proper type hints
                    schema_dict = t.cast(t.Dict[str, t.Any], schema)
                    schemas[name] = Schema(**schema_dict)
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to convert schema {name}: {e}")
                schemas[name] = Schema(
                    type="object"
                )  # Fallback to generic object schema

    # Ensure components is properly structured
    components = Components(schemas=schemas) if schemas else None

    return OpenAPISpecification(
        openapi=openapi_version,
        tags=[APP_TAG, INFRA_TAG],
        components=components,
        info=Info(
            title=svc.name,
            description=svc.doc,
            version=svc.bento.tag.version or "None" if svc.bento else "None",
            contact=Contact(name="BentoML Team", email="contact@bentoml.com"),
        ),
        servers=[{"url": "."}],
        paths={
            # setup infra endpoints
            **make_infra_endpoints(),
            # setup inference endpoints
            **_get_api_routes(svc),
            **mounted_app_paths,
        },
    )


class TaskStatusResponse(BaseModel):
    task_id: str
    status: t.Literal["in_progress", "success", "failure", "cancelled"]
    created_at: str
    executed_at: t.Optional[str]


task_status_response = {
    "description": "Successful Response",
    "content": {
        "application/json": {
            "schema": {"$ref": REF_TEMPLATE.format(model="TaskStatusResponse")}
        }
    },
}

task_status_schema = Schema(**TaskStatusResponse.model_json_schema())
error_responses = {
    ex.error_code.value: Response(
        description=field.description,
        content={
            "application/json": MediaType(
                schema=Reference(REF_TEMPLATE.format(model=field.title))
            )
        },
    )
    for ex in [InvalidArgument, NotFound, InternalServerError]
    for field in exception_schema(ex)
}


def _get_api_routes(svc: Service[t.Any]) -> dict[str, PathItem]:
    routes: dict[str, PathItem] = {}
    for api in svc.apis.values():
        post_spec = {
            "responses": {
                HTTPStatus.OK.value: api.openapi_response(),
                **error_responses,
            },
            "tags": [APP_TAG.name],
            "x-bentoml-name": api.name,
            "description": api.doc or "",
            "requestBody": api.openapi_request(),
            "operationId": f"{svc.name}__{api.name}",
        }

        # Apply any user-provided OpenAPI overrides to customize the specification
        # Users can override any valid OpenAPI operation fields like description,
        # tags, parameters, requestBody, responses, etc. The overrides are deep
        # merged with the auto-generated specification.
        if api.openapi_overrides:
            # Validate OpenAPI fields before merging
            valid_fields = {
                "description",
                "tags",
                "summary",
                "operationId",
                "parameters",
                "requestBody",
                "responses",
                "deprecated",
                "security",
                "servers",
                "externalDocs",
            }
            invalid_fields = set(api.openapi_overrides.keys()) - valid_fields
            if invalid_fields:
                raise ValueError(
                    f"Invalid OpenAPI field(s): {', '.join(invalid_fields)}. "
                    f"Valid fields are: {', '.join(sorted(valid_fields))}"
                )

            # Validate value types and schemas
            for field, value in api.openapi_overrides.items():
                if field == "parameters":
                    if not isinstance(value, list):
                        raise TypeError(
                            "Invalid value type for 'parameters'. Expected list, got "
                            f"{type(value).__name__}"
                        )
                    for param in t.cast(t.List[t.Dict[str, t.Any]], value):
                        if not isinstance(param, dict):
                            raise TypeError("Parameter must be a dictionary")
                        required_fields = {"name", "in"}
                        param_keys = set(param.keys())
                        missing = required_fields - param_keys
                        if missing:
                            raise ValueError(
                                f"Parameter missing required fields: {', '.join(missing)}"
                            )
                elif field == "responses":
                    if not isinstance(value, dict):
                        raise TypeError(
                            "Invalid value type for 'responses'. Expected dict, got "
                            f"{type(value).__name__}"
                        )
                    responses = t.cast(t.Dict[str, t.Dict[str, t.Any]], value)
                    for code, response in responses.items():
                        try:
                            code_int = int(str(code))
                            if not 100 <= code_int <= 599:
                                raise ValueError
                        except ValueError:
                            raise ValueError(
                                f"Invalid response code '{code}'. Must be a valid HTTP "
                                "status code between 100 and 599"
                            )
                        if not isinstance(response, dict):
                            raise TypeError("Response must be a dictionary")
                        if "description" not in response:
                            raise ValueError(
                                f"Response {code} missing required field: description"
                            )
                        # Validate schema types if present
                        if "content" in response:
                            for content_type, content in response["content"].items():
                                if not isinstance(content, dict):
                                    raise TypeError(
                                        f"Content for {content_type} must be a dictionary"
                                    )
                                try:
                                    content_dict: dict[str, t.Any] = {
                                        "schema": content.get("schema"),
                                        "example": content.get("example"),
                                        "examples": content.get("examples"),
                                        "encoding": content.get("encoding"),
                                    }
                                    content_obj = MediaType(**content_dict)
                                except (TypeError, ValueError) as e:
                                    raise TypeError(f"Invalid content format: {e}")

                                if content_obj.schema is not None:
                                    schema = content_obj.schema
                                    if isinstance(schema, Reference):
                                        continue
                                    valid_types = {
                                        "object",
                                        "array",
                                        "string",
                                        "number",
                                        "integer",
                                        "boolean",
                                        "null",
                                    }
                                    if (
                                        schema.type is not None
                                        and schema.type not in valid_types
                                    ):
                                        raise ValueError(
                                            f"Invalid schema type: {schema.type}. "
                                            f"Must be one of: {', '.join(sorted(valid_types))}"
                                        )
                elif field == "tags":
                    if not isinstance(value, list):
                        raise TypeError(
                            "Invalid value type for 'tags'. Expected list, got "
                            f"{type(value).__name__}"
                        )
                    tags = t.cast(t.List[str], value)
                    for tag in tags:
                        if not isinstance(tag, str):
                            raise TypeError("Tags must be strings")

            # Handle OpenAPI spec merging using proper OpenAPI types
            overrides = api.openapi_overrides
            operation_dict = dict(post_spec)  # Make a copy to avoid modifying original

            for key, value in overrides.items():
                if key == "tags" and isinstance(value, list):
                    # Merge tags list, ensuring all elements are strings
                    current_tags = t.cast(t.List[str], operation_dict.get("tags", []))
                    if not isinstance(current_tags, list):
                        current_tags = []
                    # Convert all values to strings, skip non-string values
                    value_list = t.cast(t.List[t.Any], value)
                    new_tags = [
                        str(item)
                        for item in value_list
                        if isinstance(item, (str, int, float))
                    ]
                    operation_dict["tags"] = list(set(current_tags + new_tags))
                elif key == "parameters" and isinstance(value, list):
                    # Merge parameters list with proper typing
                    current_params = t.cast(
                        t.List[t.Dict[str, t.Any]], operation_dict.get("parameters", [])
                    )
                    if not isinstance(current_params, list):
                        current_params = []
                    # Convert parameters to proper type, ensuring they're dictionaries
                    value_list = t.cast(t.List[t.Any], value)
                    param_list: t.List[t.Dict[str, t.Any]] = []
                    for param in value_list:
                        if isinstance(param, dict):
                            # Convert dictionary items with proper typing
                            param_items = t.cast(t.Dict[str, t.Any], param)
                            typed_param: t.Dict[str, t.Any] = {}
                            for k, v in param_items.items():
                                typed_param[str(k)] = v
                            param_list.append(typed_param)
                    # Merge parameters and ensure proper typing for OpenAPI spec
                    merged_params: t.List[t.Dict[str, t.Any]] = []
                    if isinstance(current_params, list):
                        merged_params.extend(current_params)
                    merged_params.extend(param_list)
                    # Convert to Operation-compatible type
                    operation_dict["parameters"] = t.cast(
                        t.Dict[str, t.Any], merged_params
                    )
                elif key == "responses" and isinstance(value, dict):
                    # Merge responses using Response type
                    current_responses = t.cast(
                        t.Dict[str, t.Union[Response, t.Dict[str, t.Any]]],
                        operation_dict.get("responses", {}),
                    )
                    if not isinstance(current_responses, dict):
                        current_responses = {}
                    merged_responses: t.Dict[str, Response] = {}

                    # Helper function to create MediaType objects safely
                    def create_media_type(spec: t.Dict[str, t.Any]) -> MediaType:
                        return MediaType(
                            schema=spec.get("schema"),
                            example=spec.get("example"),
                            examples=spec.get("examples"),
                            encoding=spec.get("encoding"),
                        )

                    # Helper function to create Response objects safely
                    def create_response(resp_dict: t.Dict[str, t.Any]) -> Response:
                        if "description" not in resp_dict:
                            resp_dict = dict(resp_dict)
                            resp_dict["description"] = "Response"
                        if "content" in resp_dict:
                            content: t.Dict[str, MediaType] = {}
                            content_dict = t.cast(
                                t.Dict[str, t.Dict[str, t.Any]], resp_dict["content"]
                            )
                            for content_type, content_spec in content_dict.items():
                                if isinstance(content_spec, dict):
                                    content[content_type] = create_media_type(
                                        content_spec
                                    )
                            resp_dict["content"] = content
                        return Response(**resp_dict)

                    # First, convert existing responses to proper Response objects
                    for code, resp in current_responses.items():
                        if isinstance(resp, dict):
                            merged_responses[str(code)] = create_response(resp)
                        elif isinstance(resp, Response):
                            merged_responses[str(code)] = resp

                    # Then merge in new responses
                    response_dict = t.cast(
                        t.Dict[str, t.Union[Response, t.Dict[str, t.Any]]], value
                    )
                    for code, resp in response_dict.items():
                        if isinstance(resp, dict):
                            merged_responses[str(code)] = create_response(resp)
                        elif isinstance(resp, Response):
                            merged_responses[str(code)] = resp
                    operation_dict["responses"] = merged_responses
                else:
                    # For other fields (description, summary, etc.), just override
                    operation_dict[key] = value

            # Create Operation object with merged data
            # Don't clear post_spec, just update it with merged data
            post_spec.update(operation_dict)

        routes[api.route] = PathItem(post=post_spec)
        if api.is_task:
            routes[f"{api.route}/status"] = PathItem(
                get={
                    "responses": {
                        HTTPStatus.OK.value: task_status_response,
                        **error_responses,
                    },
                    "tags": [APP_TAG.name],
                    "x-bentoml-name": f"{api.name}_status",
                    "description": f"Get status of task {api.name}",
                    "operationId": f"{svc.name}__{api.name}_status",
                    "parameters": [
                        {
                            "name": "task_id",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string", "title": "Task ID"},
                        }
                    ],
                }
            )
            routes[f"{api.route}/get"] = PathItem(
                get={
                    "responses": {
                        HTTPStatus.OK.value: api.openapi_response(),
                        **error_responses,
                    },
                    "tags": [APP_TAG.name],
                    "x-bentoml-name": f"{api.name}_result",
                    "description": f"Get result of task {api.name}",
                    "operationId": f"{svc.name}__{api.name}_result",
                    "parameters": [
                        {
                            "name": "task_id",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string", "title": "Task ID"},
                        }
                    ],
                }
            )
            routes[f"{api.route}/submit"] = PathItem(
                post={
                    "responses": {
                        HTTPStatus.OK.value: task_status_response,
                        **error_responses,
                    },
                    "tags": [APP_TAG.name],
                    "x-bentoml-name": f"{api.name}_submit",
                    "description": f"Submit a new task of {api.name}",
                    "operationId": f"{svc.name}__{api.name}_submit",
                    "requestBody": api.openapi_request(),
                }
            )
            routes[f"{api.route}/retry"] = PathItem(
                post={
                    "responses": {
                        HTTPStatus.OK.value: task_status_response,
                        **error_responses,
                    },
                    "tags": [APP_TAG.name],
                    "x-bentoml-name": f"{api.name}_retry",
                    "description": f"Retry a task of {api.name}",
                    "operationId": f"{svc.name}__{api.name}_retry",
                    "parameters": [
                        {
                            "name": "task_id",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string", "title": "Task ID"},
                        }
                    ],
                }
            )
            routes[f"{api.route}/cancel"] = PathItem(
                put={
                    "responses": {
                        HTTPStatus.OK.value: task_status_response,
                        **error_responses,
                    },
                    "tags": [APP_TAG.name],
                    "x-bentoml-name": f"{api.name}_retry",
                    "description": f"Cancel an in-progress task of {api.name}",
                    "operationId": f"{svc.name}__{api.name}_retry",
                    "parameters": [
                        {
                            "name": "task_id",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string", "title": "Task ID"},
                        }
                    ],
                }
            )
    return routes


def generate_service_components(svc: Service[t.Any]) -> dict[str, t.Any]:
    components: dict[str, Schema] = {}
    for name, api in svc.apis.items():
        input_components = api.input_spec.openapi_components(name)
        components.update(input_components)
        output_components = api.output_spec.openapi_components(name)
        components.update(output_components)

    components["TaskStatusResponse"] = task_status_schema
    # merge exception at last
    components.update(exception_components_schema())
    return {"schemas": components}
