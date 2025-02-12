from __future__ import annotations

import typing as t

import attr
import yaml

from ..utils.cattr import bentoml_cattr
from .schemas.schemasv1 import CreateSecretSchema
from .schemas.schemasv1 import SecretContentSchema
from .schemas.schemasv1 import SecretItem
from .schemas.schemasv1 import SecretSchema
from .schemas.schemasv1 import UpdateSecretSchema

if t.TYPE_CHECKING:
    from .client import RestApiClient


@attr.define
class Secret(SecretSchema):
    created_by: str

    def to_dict(self) -> dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)

    def to_yaml(self):
        dt = self.to_dict()
        return yaml.dump(dt, sort_keys=False)

    @classmethod
    def from_secret_schema(cls, secret_schema: SecretSchema) -> Secret:
        return cls(
            name=secret_schema.name,
            uid=secret_schema.uid,
            resource_type=secret_schema.resource_type,
            labels=secret_schema.labels,
            created_at=secret_schema.created_at,
            updated_at=secret_schema.updated_at,
            deleted_at=secret_schema.deleted_at,
            created_by=secret_schema.creator.name,
            description=secret_schema.description,
            creator=secret_schema.creator,
            content=secret_schema.content,
            cluster=secret_schema.cluster,
        )


@attr.define
class SecretAPI:
    _client: RestApiClient

    def list(self, search: str | None = None) -> t.List[Secret]:
        """
        List all secrets.

        Args:
            search (str | None): Optional search string to filter secrets.

        Returns:
            List[SecretInfo]: A list of SecretInfo objects representing the secrets.
        """
        secrets = self._client.v1.list_secrets(search=search)
        return [Secret.from_secret_schema(secret) for secret in secrets.items]

    def create(
        self,
        name: str,
        type: str,
        cluster: str | None = None,
        description: str | None = None,
        path: str | None = None,
        key_vals: t.List[t.Tuple[str, str]] = [],
    ) -> Secret:
        """
        Create a new secret.

        Args:
            name (str | None): Name of the secret.
            description (str | None): Description of the secret.
            type (str | None): Type of the secret.
            cluster (str | None): Cluster name where the secret is created.
            path (str | None): Path of the secret.
            key_vals (List[Tuple[str, str]]): List of key-value pairs for the secret content.

        Returns:
            SecretInfo: A SecretInfo object representing the created secret.
        """
        secret_schema = CreateSecretSchema(
            name=name,
            content=SecretContentSchema(
                type=type,
                path=path,
                items=[
                    SecretItem(key=key_val[0], value=key_val[1]) for key_val in key_vals
                ],
            ),
            description=description,
        )
        secret = self._client.v1.create_secret(secret_schema, cluster)
        return Secret.from_secret_schema(secret)

    def delete(self, name: str | None = None, cluster: str | None = None):
        """
        Delete a secret.

        Args:
            name (str | None): Name of the secret to delete.

        Raises:
            ValueError: If name is None.
        """
        if name is None:
            raise ValueError("name is required")
        self._client.v1.delete_secret(name, cluster)

    def update(
        self,
        name: str,
        type: str,
        cluster: str | None = None,
        description: str | None = None,
        path: str | None = None,
        key_vals: t.List[t.Tuple[str, str]] = [],
    ) -> Secret:
        """
        Update an existing secret.

        Args:
            name (str | None): Name of the secret to update.
            description (str | None): New description of the secret.
            type (str | None): New type of the secret.
            path (str | None): New path of the secret.
            key_vals (List[Tuple[str, str]]): New list of key-value pairs for the secret content.

        Returns:
            SecretInfo: A SecretInfo object representing the updated secret.
        """
        secret_schema = UpdateSecretSchema(
            content=SecretContentSchema(
                type=type,
                path=path,
                items=[
                    SecretItem(key=key_val[0], value=key_val[1]) for key_val in key_vals
                ],
            ),
            description=description,
        )
        secret = self._client.v1.update_secret(name, secret_schema, cluster)
        return Secret.from_secret_schema(secret)

    def get(self, name: str, cluster: str | None = None) -> Secret:
        """
        Get a secret by name.
        """
        secret = self._client.v1.get_secret(name, cluster)
        return Secret.from_secret_schema(secret)
