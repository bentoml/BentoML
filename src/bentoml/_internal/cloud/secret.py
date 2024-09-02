from __future__ import annotations

import typing as t

import attr
import yaml
from simple_di import Provide
from simple_di import inject

from ..configuration.containers import BentoMLContainer
from ..utils import bentoml_cattr
from .schemas.schemasv1 import CreateSecretSchema
from .schemas.schemasv1 import SecretContentSchema
from .schemas.schemasv1 import SecretItem
from .schemas.schemasv1 import SecretSchema
from .schemas.schemasv1 import UpdateSecretSchema

if t.TYPE_CHECKING:
    from .client import RestApiClient


@attr.define
class SecretInfo(SecretSchema):
    created_by: str

    def to_dict(self) -> dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)

    def to_yaml(self):
        dt = self.to_dict()
        return yaml.dump(dt, sort_keys=False)

    @classmethod
    def from_secret_schema(cls, secret_schema: SecretSchema) -> SecretInfo:
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
        )


@attr.define
class Secret:
    @classmethod
    @inject
    def list(
        cls,
        search: str | None = None,
        cloud_rest_client: RestApiClient = Provide[BentoMLContainer.rest_api_client],
    ) -> t.List[SecretInfo]:
        secrets = cloud_rest_client.v1.list_secrets(search=search)
        return [SecretInfo.from_secret_schema(secret) for secret in secrets.items]

    @classmethod
    @inject
    def create(
        cls,
        name: str | None = None,
        description: str | None = None,
        type: str | None = None,
        path: str | None = None,
        key_vals: t.List[t.Tuple[str, str]] = [],
        cloud_rest_client: RestApiClient = Provide[BentoMLContainer.rest_api_client],
    ) -> SecretInfo:
        secret_schema = CreateSecretSchema(
            name=name,
            description=description,
            content=SecretContentSchema(
                type=type,
                path=path,
                items=[
                    SecretItem(key=key_val[0], value=key_val[1]) for key_val in key_vals
                ],
            ),
        )
        secret = cloud_rest_client.v1.create_secret(secret_schema)
        return SecretInfo.from_secret_schema(secret)

    @classmethod
    @inject
    def delete(
        cls,
        name: str | None = None,
        cloud_rest_client: RestApiClient = Provide[BentoMLContainer.rest_api_client],
    ):
        if name is None:
            raise ValueError("name is required")
        cloud_rest_client.v1.delete_secret(name)

    @classmethod
    @inject
    def update(
        cls,
        name: str | None = None,
        description: str | None = None,
        type: str | None = None,
        path: str | None = None,
        key_vals: t.List[t.Tuple[str, str]] = [],
        cloud_rest_client: RestApiClient = Provide[BentoMLContainer.rest_api_client],
    ) -> SecretInfo:
        secret_schema = UpdateSecretSchema(
            description=description,
            content=SecretContentSchema(
                type=type,
                path=path,
                items=[
                    SecretItem(key=key_val[0], value=key_val[1]) for key_val in key_vals
                ],
            ),
        )
        secret = cloud_rest_client.v1.update_secret(name, secret_schema)
        return SecretInfo.from_secret_schema(secret)
