from __future__ import annotations

import typing as t
from datetime import datetime

import attr
import yaml

from ..utils.cattr import bentoml_cattr
from .schemas.schemasv1 import ApiTokenSchema
from .schemas.schemasv1 import CreateApiTokenSchema

if t.TYPE_CHECKING:
    from .client import RestApiClient


@attr.define(kw_only=True)
class ApiToken(ApiTokenSchema):
    created_by: str = attr.field(default="")

    def to_dict(self) -> dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)

    def to_yaml(self) -> str:
        dt = self.to_dict()
        return yaml.dump(dt, sort_keys=False)

    @classmethod
    def from_schema(cls, schema: ApiTokenSchema) -> ApiToken:
        return cls(
            name=schema.name,
            uid=schema.uid,
            resource_type=schema.resource_type,
            labels=schema.labels,
            created_at=schema.created_at,
            updated_at=schema.updated_at,
            deleted_at=schema.deleted_at,
            description=schema.description,
            scopes=schema.scopes,
            user=schema.user,
            organization=schema.organization,
            expired_at=schema.expired_at,
            last_used_at=schema.last_used_at,
            is_expired=schema.is_expired,
            is_api_token=schema.is_api_token,
            is_organization_token=schema.is_organization_token,
            is_global_access=schema.is_global_access,
            token=schema.token,
            created_by=schema.user.name,
        )


@attr.define
class ApiTokenAPI:
    _client: RestApiClient

    def list(self, search: str | None = None) -> t.List[ApiToken]:
        """
        List all API tokens.

        Args:
            search (str | None): Optional search string to filter tokens.

        Returns:
            List[ApiToken]: A list of ApiToken objects.
        """
        tokens = self._client.v1.list_api_tokens(search=search)
        return [ApiToken.from_schema(token) for token in tokens.items]

    def create(
        self,
        name: str,
        description: str | None = None,
        scopes: t.List[str] | None = None,
        expired_at: datetime | None = None,
    ) -> ApiToken:
        """
        Create a new API token.

        Args:
            name (str): Name of the token.
            description (str | None): Description of the token.
            scopes (List[str] | None): List of scopes for the token.
            expired_at (datetime | None): Expiration datetime for the token.

        Returns:
            ApiToken: The created API token (includes the token value).
        """
        schema = CreateApiTokenSchema(
            name=name,
            description=description,
            scopes=scopes,
            expired_at=expired_at,
        )
        token = self._client.v1.create_api_token(schema)
        return ApiToken.from_schema(token)

    def get(self, token_uid: str) -> ApiToken | None:
        """
        Get an API token by UID.

        Args:
            token_uid (str): The UID of the token to get.

        Returns:
            ApiToken | None: The API token if found, None otherwise.
        """
        token = self._client.v1.get_api_token(token_uid)
        if token is None:
            return None
        return ApiToken.from_schema(token)

    def delete(self, token_uid: str) -> None:
        """
        Delete an API token.

        Args:
            token_uid (str): The UID of the token to delete.
        """
        self._client.v1.delete_api_token(token_uid)
