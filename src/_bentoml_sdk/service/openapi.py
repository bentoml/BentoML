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
from bentoml._internal.service.openapi.specification import RequestBody
from bentoml._internal.service.openapi.specification import Response
from bentoml._internal.service.openapi.specification import Schema
from bentoml._internal.service.openapi.specification import Tag
from bentoml._internal.service.openapi.utils import exception_components_schema
from bentoml._internal.service.openapi.utils import exception_schema
from bentoml._internal.types import LazyType
from bentoml._internal.utils import bentoml_cattr
from bentoml.exceptions import InternalServerError
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import NotFound

# Type aliases for better type safety
OperationValue = t.Union[str, int, float, bool, None, dict[str, t.Any], list[t.Any]]
OperationFieldType = t.Union[
    str,
    bool,
    list[t.Union[str, dict[str, t.Any]]],  # For tags, parameters, security
    dict[str, t.Any],  # For externalDocs, requestBody
    RequestBody,  # For typed requestBody
]
ValidatedDict = t.Dict[str, OperationFieldType]

# Type mapping for validation
OperationFieldTypes = t.Dict[str, t.Union[type, tuple[type, ...]]]

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


def generate_spec(
    svc: Service[t.Any], *, openapi_version: str = "3.0.2"
) -> OpenAPISpecification:
    """Generate an OpenAPI specification for a service.

    Args:
        svc: The BentoML service to generate the specification for
        openapi_version: The OpenAPI specification version to use

    Returns:
        An OpenAPI specification for the service
    """
    # Initialize paths and components
    mounted_app_paths: Dict[str, PathItem] = {}
    schema_components: Dict[str, Dict[str, SchemaType]] = {}

    # Define valid component types
    VALID_COMPONENT_TYPES = frozenset(
        [
            "schemas",
            "responses",
            "examples",
            "requestBodies",
            "headers",
            "securitySchemes",
            "links",
            "callbacks",
        ]
    )

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

    # Apply service-level OpenAPI overrides if present
    if svc.openapi_service_overrides:
        overrides = svc.openapi_service_overrides
        if "components" in overrides:
            merger.merge(schema_components, overrides["components"])

    # Convert schema components to proper type, avoiding recursion
    def convert_schema(
        schema_data: t.Union[Schema, Reference, dict[str, t.Any]], depth: int = 0
    ) -> t.Union[Schema, Reference]:
        """Convert a schema input to a proper Schema or Reference object.

        Args:
            schema_data: Input schema data (Schema, Reference, or dict)
            depth: Current recursion depth for preventing infinite loops

        Returns:
            Schema or Reference object
        """
        MAX_DEPTH = 5  # Reduced depth limit to prevent recursion errors

        if depth > MAX_DEPTH:
            return Schema(type="object")
        if isinstance(schema_data, (Schema, Reference)):
            return schema_data
        if not isinstance(schema_data, dict):
            return Schema(type="object")

        schema_dict = dict(schema_data)
        if "$ref" in schema_dict:
            return Reference(ref=str(schema_dict["$ref"]))

        # Handle nested schemas in properties
        if "properties" in schema_dict:
            props_dict = t.cast(
                dict[str, dict[str, t.Any]], schema_dict.get("properties", {})
            )
            if isinstance(props_dict, dict):
                converted_props: dict[str, t.Union[Schema, Reference]] = {}

                for name, prop_data in props_dict.items():
                    prop_key = str(name)
                    if isinstance(prop_data, (dict, Schema, Reference)):
                        converted_props[prop_key] = convert_schema(prop_data, depth + 1)
                    else:
                        converted_props[prop_key] = Schema(type="object")

                schema_dict["properties"] = converted_props

        # Handle array items
        if "items" in schema_dict:
            items_data = schema_dict.get("items")
            if isinstance(items_data, (dict, Schema, Reference)):
                items_schema = t.cast(
                    t.Union[Schema, Reference, dict[str, t.Any]], items_data
                )
                schema_dict["items"] = convert_schema(items_schema, depth + 1)
            elif isinstance(items_data, list):
                items_list = t.cast(
                    list[t.Union[dict[str, t.Any], Schema, Reference]], items_data
                )
                converted_items: list[t.Union[Schema, Reference]] = []

                for item_data in items_list:
                    if isinstance(item_data, (dict, Schema, Reference)):
                        converted_items.append(convert_schema(item_data, depth + 1))
                    else:
                        converted_items.append(Schema(type="object"))

                schema_dict["items"] = converted_items

        try:
            return Schema(**schema_dict)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to convert schema: {e}")
            return Schema(type="object")

    schemas: dict[str, t.Union[Schema, Reference]] = {}
    schema_dict = schema_components.get("schemas", {})
    for name, schema in schema_dict.items():
        schemas[name] = convert_schema(schema)

    # Ensure components is properly structured
    components = Components(schemas=schemas) if schemas else None

    # Prepare base specification
    info = Info(
        title=svc.name,
        description=svc.doc,
        version=svc.bento.tag.version or "None" if svc.bento else "None",
        contact=Contact(name="BentoML Team", email="contact@bentoml.com"),
    )

    # Apply service-level overrides to Info object
    if svc.openapi_service_overrides:
        overrides = svc.openapi_service_overrides
        # Validate and apply Info fields with proper type checking
        try:
            title = overrides.get("title")
            if title is not None:
                if not isinstance(title, (str, int, float)):
                    logger.warning(
                        f"Invalid type for OpenAPI title override: {type(title)}. "
                        "Using service name as fallback."
                    )
                else:
                    title = str(title)

            description = overrides.get("description")
            if description is not None:
                if not isinstance(description, (str, int, float)):
                    logger.warning(
                        f"Invalid type for OpenAPI description override: {type(description)}. "
                        "Using service doc as fallback."
                    )
                    description = info.description
                else:
                    description = str(description)

            version = overrides.get("version")
            if version is not None:
                if not isinstance(version, (str, int, float)):
                    logger.warning(
                        f"Invalid type for OpenAPI version override: {type(version)}. "
                        "Using service version as fallback."
                    )
                    version = info.version
                else:
                    version = str(version)

            # Create new Info object with validated overrides
            info = Info(
                title=title or info.title,
                description=description or info.description or "",
                version=version or info.version,
                contact=info.contact,
            )
        except (TypeError, ValueError) as e:
            logger.warning(
                f"Error applying OpenAPI info overrides: {e}. Using defaults."
            )

    # Initialize components with proper validation
    components = Components(schemas={})
    if svc.openapi_service_overrides and "components" in svc.openapi_service_overrides:
        override_components = t.cast(
            t.Dict[str, t.Any], svc.openapi_service_overrides.get("components", {})
        )
        if isinstance(override_components, dict):
            try:
                # Handle schemas
                if "schemas" in override_components:
                    schema_dict = t.cast(
                        t.Dict[str, t.Any], override_components.get("schemas", {})
                    )
                    if isinstance(schema_dict, dict):
                        for name, schema in schema_dict.items():
                            try:
                                if isinstance(schema, dict):
                                    components.schemas[str(name)] = Schema(
                                        **t.cast(t.Dict[str, t.Any], schema)
                                    )
                                elif isinstance(schema, (Schema, Reference)):
                                    components.schemas[str(name)] = schema
                                else:
                                    logger.warning(
                                        f"Invalid schema type for '{name}': {type(schema).__name__}"
                                    )
                            except (TypeError, ValueError) as e:
                                logger.warning(f"Invalid schema '{name}': {e}")

                # Handle other component types
                for comp_type in VALID_COMPONENT_TYPES:
                    if comp_type in override_components:
                        comp_value = override_components.get(comp_type)
                        if isinstance(comp_value, dict):
                            # Create a new dictionary to avoid modifying the original
                            validated_comp = {
                                str(k): v
                                for k, v in t.cast(
                                    t.Dict[str, t.Any], comp_value
                                ).items()
                            }
                            setattr(components, comp_type, validated_comp)
                        else:
                            logger.warning(
                                f"Invalid {comp_type} type: {type(comp_value).__name__ if comp_value is not None else 'None'}. Expected dict."
                            )
            except Exception as e:
                logger.warning(f"Error merging components: {e}")
        else:
            logger.warning(
                f"Invalid components override type: {type(override_components).__name__}. Expected dict."
            )

    # Prepare info object with proper validation
    title = svc.name
    description = svc.doc or ""
    version = svc.bento.tag.version or "None" if svc.bento else "None"

    # Apply overrides with validation
    if svc.openapi_service_overrides:
        if "title" in svc.openapi_service_overrides:
            title_override = svc.openapi_service_overrides["title"]
            if isinstance(title_override, (str, int, float)):
                title = str(title_override)
            else:
                logger.warning(
                    f"Invalid title override type: {type(title_override).__name__}. Using service name."
                )

        if "description" in svc.openapi_service_overrides:
            desc_override = svc.openapi_service_overrides["description"]
            if isinstance(desc_override, (str, int, float)):
                description = str(desc_override)
            else:
                logger.warning(
                    f"Invalid description override type: {type(desc_override).__name__}. Using service doc."
                )

        if "version" in svc.openapi_service_overrides:
            version_override = svc.openapi_service_overrides["version"]
            if isinstance(version_override, (str, int, float)):
                version = str(version_override)
            else:
                logger.warning(
                    f"Invalid version override type: {type(version_override).__name__}. Using service version."
                )

    # Create base specification
    spec = OpenAPISpecification(
        openapi=openapi_version,
        tags=[APP_TAG, INFRA_TAG],
        components=components,
        info=Info(
            title=title,
            description=description,
            version=version,
            contact=Contact(name="BentoML Team", email="contact@bentoml.com"),
        ),
        servers=[{"url": "."}],  # Default server
        paths={
            **make_infra_endpoints(),
            **_get_api_routes(svc),
            **mounted_app_paths,
        },
    )

    # Handle server overrides with proper type checking and error handling
    if svc.openapi_service_overrides and "servers" in svc.openapi_service_overrides:
        try:
            servers_raw = svc.openapi_service_overrides.get("servers", [])
            if not isinstance(servers_raw, list):
                logger.warning(
                    f"Invalid servers override type: expected list, got {type(servers_raw).__name__}"
                )
                return spec

            # Process each server entry
            validated_servers: list[dict[str, t.Any]] = []
            server_entries = t.cast(list[dict[str, t.Any]], servers_raw)

            for server_entry in server_entries:
                if not isinstance(server_entry, dict):
                    logger.warning(
                        f"Invalid server object type: expected dict, got {type(server_entry).__name__}"
                    )
                    continue

                if "url" not in server_entry:
                    logger.warning("Server object missing required 'url' field")
                    continue

                try:
                    server_data: dict[str, t.Any] = {"url": str(server_entry["url"])}

                    # Validate optional fields
                    if "description" in server_entry:
                        desc = server_entry["description"]
                        if isinstance(desc, (str, int, float, bool)):
                            server_data["description"] = str(desc)

                    # Validate variables if present
                    if "variables" in server_entry:
                        vars_dict = t.cast(
                            dict[str, dict[str, t.Any]],
                            server_entry.get("variables", {}),
                        )
                        if isinstance(vars_dict, dict):
                            validated_vars: dict[str, dict[str, str]] = {}

                            for var_name, var_def in vars_dict.items():
                                if isinstance(var_def, dict):
                                    var_obj: dict[str, str] = {}

                                    if "default" in var_def:
                                        default_val = var_def.get("default")
                                        if default_val is not None:
                                            var_obj["default"] = str(default_val)

                                    if "enum" in var_def:
                                        enum_vals = t.cast(
                                            list[t.Any], var_def.get("enum", [])
                                        )
                                        if isinstance(enum_vals, list):
                                            var_obj["enum"] = (
                                                "["
                                                + ", ".join(
                                                    str(val) for val in enum_vals
                                                )
                                                + "]"
                                            )

                                    if "description" in var_def:
                                        desc_val = var_def.get("description")
                                        if desc_val is not None:
                                            var_obj["description"] = str(desc_val)

                                    if var_obj:
                                        validated_vars[str(var_name)] = var_obj

                            if validated_vars:
                                server_data["variables"] = validated_vars

                    validated_servers.append(server_data)
                except (TypeError, ValueError, KeyError) as err:
                    logger.warning(f"Error validating server object: {err}")

            if validated_servers:
                spec.servers = validated_servers
        except Exception as e:
            logger.warning(f"Error processing server overrides: {e}")

    # Handle tags with proper validation
    if svc.openapi_service_overrides and "tags" in svc.openapi_service_overrides:
        tag_list = t.cast(
            list[t.Union[dict[str, t.Any], Tag]], svc.openapi_service_overrides["tags"]
        )
        if isinstance(tag_list, list):
            # Ensure we keep the required APP_TAG and INFRA_TAG
            validated_tags: list[Tag] = []
            for tag_item in tag_list:
                if isinstance(tag_item, dict) and "name" in tag_item:
                    try:
                        validated_tags.append(Tag(**tag_item))
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Invalid tag definition: {e}")
                        continue
                elif isinstance(tag_item, Tag):
                    validated_tags.append(tag_item)
            if validated_tags:
                spec.tags = [APP_TAG, INFRA_TAG] + validated_tags

    return spec


class TaskStatusResponse(BaseModel):  # type: ignore[misc]
    """Response model for task status endpoint."""

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
        # Start with service-level overrides if they exist
        post_spec = {}
        if svc.openapi_service_overrides:
            # Apply service-level operation fields that make sense at the endpoint level
            valid_operation_fields: OperationFieldTypes = {
                "description": str,
                "summary": str,
                "tags": list,
                "externalDocs": dict,
                "parameters": list,
                "requestBody": (dict, RequestBody),
                "responses": dict,
                "deprecated": bool,
                "security": list,
                "servers": list,
            }

            # Validate and convert operation fields
            validated_overrides: ValidatedDict = {}
            for field, expected_type in valid_operation_fields.items():
                if field not in svc.openapi_service_overrides:
                    continue

                value = svc.openapi_service_overrides[field]
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        continue
                elif not isinstance(value, expected_type):
                    # Try to convert basic types
                    try:
                        if expected_type is str:
                            value = str(value)
                        elif expected_type is bool:
                            value = bool(value)
                        elif expected_type is list and isinstance(value, (tuple, set)):
                            value = list(value)
                        elif expected_type is dict and hasattr(value, "__dict__"):
                            value = dict(value.__dict__)
                        else:
                            continue
                    except (ValueError, TypeError):
                        continue

                validated_overrides[field] = value

            if validated_overrides:
                post_spec.update(validated_overrides)

        # Add/override with standard fields
        post_spec.update(
            {
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
        )

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
            # Create a fresh dictionary to avoid reference issues
            post_spec = {**operation_dict}

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
