from __future__ import annotations

import contextlib
import logging
import threading
import typing as t
import uuid
from queue import Empty
from urllib.parse import urlencode

import httpx
from httpx_ws import WebSocketNetworkError
from httpx_ws import connect_ws
from wsproto.utilities import LocalProtocolError

from ...exceptions import CloudRESTApiClientError
from ...exceptions import NotFound
from ..configuration import BENTOML_VERSION
from .schemas.schemasv1 import BentoListSchema
from .schemas.schemasv1 import BentoRepositorySchema
from .schemas.schemasv1 import BentoSchema
from .schemas.schemasv1 import BentoWithRepositoryListSchema
from .schemas.schemasv1 import ClusterFullSchema
from .schemas.schemasv1 import ClusterListSchema
from .schemas.schemasv1 import CompleteMultipartUploadSchema
from .schemas.schemasv1 import CreateBentoRepositorySchema
from .schemas.schemasv1 import CreateBentoSchema
from .schemas.schemasv1 import CreateDeploymentSchema as CreateDeploymentSchemaV1
from .schemas.schemasv1 import CreateModelRepositorySchema
from .schemas.schemasv1 import CreateModelSchema
from .schemas.schemasv1 import CreateSecretSchema
from .schemas.schemasv1 import DeploymentFullSchema
from .schemas.schemasv1 import DeploymentListSchema
from .schemas.schemasv1 import FinishUploadSchema
from .schemas.schemasv1 import ModelRepositorySchema
from .schemas.schemasv1 import ModelSchema
from .schemas.schemasv1 import ModelWithRepositoryListSchema
from .schemas.schemasv1 import OrganizationSchema
from .schemas.schemasv1 import PreSignMultipartUploadUrlSchema
from .schemas.schemasv1 import ResourceInstanceSchema
from .schemas.schemasv1 import SecretListSchema
from .schemas.schemasv1 import SecretSchema
from .schemas.schemasv1 import UpdateBentoSchema
from .schemas.schemasv1 import UpdateDeploymentSchema
from .schemas.schemasv1 import UpdateSecretSchema
from .schemas.schemasv1 import UserSchema
from .schemas.schemasv2 import CreateDeploymentSchema as CreateDeploymentSchemaV2
from .schemas.schemasv2 import DeleteDeploymentFilesSchema
from .schemas.schemasv2 import DeploymentFileListSchema
from .schemas.schemasv2 import DeploymentFullSchema as DeploymentFullSchemaV2
from .schemas.schemasv2 import DeploymentListSchema as DeploymentListSchemaV2
from .schemas.schemasv2 import KubePodSchema
from .schemas.schemasv2 import KubePodWSResponseSchema
from .schemas.schemasv2 import LogWSResponseSchema
from .schemas.schemasv2 import UpdateDeploymentSchema as UpdateDeploymentSchemaV2
from .schemas.schemasv2 import UploadDeploymentFilesSchema
from .schemas.utils import schema_from_object
from .schemas.utils import schema_to_object

logger = logging.getLogger(__name__)


class BaseRestApiClient:
    def __init__(self, session: httpx.Client) -> None:
        self.session = session

    @staticmethod
    def _is_not_found(resp: httpx.Response) -> bool:
        # We used to return 400 for record not found, handle both cases
        return (
            resp.status_code == 404
            or resp.status_code == 400
            and "record not found" in resp.text
        )

    @staticmethod
    def _check_resp(resp: httpx.Response) -> None:
        if resp.status_code >= 500:
            if "x-trace-id" in resp.headers:
                logger.error(
                    "request failed with status code %d: %s\n\n",
                    resp.status_code,
                    resp.text,
                )
                raise CloudRESTApiClientError(
                    f"Oops, something went wrong with BentoCloud. Please report to us with trace ID <{resp.headers['x-trace-id']}>: https://bentoml.com/support"
                )
            else:
                raise CloudRESTApiClientError(
                    f"request failed with status code {resp.status_code}: {resp.text}",
                    error_code=resp.status_code,
                )
        if resp.status_code != 200:
            raise CloudRESTApiClientError(
                f"request failed with status code {resp.status_code}: {resp.text}",
                error_code=resp.status_code,
            )


class RestApiClientV1(BaseRestApiClient):
    def get_current_user(self) -> UserSchema | None:
        url = "/api/v1/auth/current"
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), UserSchema)

    def get_current_organization(self) -> OrganizationSchema | None:
        url = "/api/v1/current_org"
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), OrganizationSchema)

    def get_bento_repository(
        self, bento_repository_name: str
    ) -> BentoRepositorySchema | None:
        url = f"/api/v1/bento_repositories/{bento_repository_name}"
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoRepositorySchema)

    def create_bento_repository(
        self, req: CreateBentoRepositorySchema
    ) -> BentoRepositorySchema:
        url = "/api/v1/bento_repositories"
        resp = self.session.post(url, json=schema_to_object(req))
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoRepositorySchema)

    def get_bento(self, bento_repository_name: str, version: str) -> BentoSchema | None:
        url = f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}"
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoSchema)

    def list_bentos(
        self, bento_repository_name: str, count: int = 1
    ) -> BentoListSchema | None:
        url = f"/api/v1/bento_repositories/{bento_repository_name}/bentos"
        resp = self.session.get(url, params={"count": count})
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoListSchema)

    def create_bento(
        self, bento_repository_name: str, req: CreateBentoSchema
    ) -> BentoSchema:
        url = f"/api/v1/bento_repositories/{bento_repository_name}/bentos"
        resp = self.session.post(url, json=schema_to_object(req))
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoSchema)

    def update_bento(
        self, bento_repository_name: str, version: str, req: UpdateBentoSchema
    ) -> BentoSchema:
        url = f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}"
        resp = self.session.patch(url, json=schema_to_object(req))
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoSchema)

    def presign_bento_upload_url(
        self, bento_repository_name: str, version: str
    ) -> BentoSchema:
        url = f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}/presign_upload_url"
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoSchema)

    def presign_bento_download_url(
        self, bento_repository_name: str, version: str
    ) -> BentoSchema:
        url = f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}/presign_download_url"
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoSchema)

    def start_bento_multipart_upload(
        self, bento_repository_name: str, version: str
    ) -> BentoSchema:
        url = f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}/start_multipart_upload"
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoSchema)

    def presign_bento_multipart_upload_url(
        self,
        bento_repository_name: str,
        version: str,
        req: PreSignMultipartUploadUrlSchema,
    ) -> BentoSchema:
        url = f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}/presign_multipart_upload_url"
        resp = self.session.patch(url, json=schema_to_object(req))
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoSchema)

    def complete_bento_multipart_upload(
        self,
        bento_repository_name: str,
        version: str,
        req: CompleteMultipartUploadSchema,
    ) -> BentoSchema:
        url = f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}/complete_multipart_upload"
        resp = self.session.patch(url, json=schema_to_object(req))
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoSchema)

    def start_upload_bento(
        self, bento_repository_name: str, version: str
    ) -> BentoSchema:
        url = f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}/start_upload"
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoSchema)

    def finish_upload_bento(
        self, bento_repository_name: str, version: str, req: FinishUploadSchema
    ) -> BentoSchema:
        url = f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}/finish_upload"
        resp = self.session.patch(url, json=schema_to_object(req))
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoSchema)

    def upload_bento(
        self, bento_repository_name: str, version: str, data: t.IO[bytes]
    ) -> None:
        url = f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}/upload"
        resp = self.session.put(
            url,
            content=data,
            headers=dict(
                self.session.headers, **{"Content-Type": "application/octet-stream"}
            ),
        )
        self._check_resp(resp)
        return None

    @contextlib.contextmanager
    def download_bento(
        self, bento_repository_name: str, version: str
    ) -> t.Generator[httpx.Response, None, None]:
        url = f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}/download"
        with self.session.stream("GET", url) as resp:
            self._check_resp(resp)
            yield resp

    def get_model_repository(
        self, model_repository_name: str
    ) -> ModelRepositorySchema | None:
        url = f"/api/v1/model_repositories/{model_repository_name}"
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), ModelRepositorySchema)

    def create_model_repository(
        self, req: CreateModelRepositorySchema
    ) -> ModelRepositorySchema:
        url = "/api/v1/model_repositories"
        resp = self.session.post(url, json=schema_to_object(req))
        self._check_resp(resp)
        return schema_from_object(resp.json(), ModelRepositorySchema)

    def get_model(self, model_repository_name: str, version: str) -> ModelSchema | None:
        url = f"/api/v1/model_repositories/{model_repository_name}/models/{version}"
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), ModelSchema)

    def create_model(
        self, model_repository_name: str, req: CreateModelSchema
    ) -> ModelSchema:
        url = f"/api/v1/model_repositories/{model_repository_name}/models"
        resp = self.session.post(url, json=schema_to_object(req))
        self._check_resp(resp)
        return schema_from_object(resp.json(), ModelSchema)

    def presign_model_upload_url(
        self, model_repository_name: str, version: str
    ) -> ModelSchema:
        url = f"/api/v1/model_repositories/{model_repository_name}/models/{version}/presign_upload_url"
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_object(resp.json(), ModelSchema)

    def presign_model_download_url(
        self, model_repository_name: str, version: str
    ) -> ModelSchema:
        url = f"/api/v1/model_repositories/{model_repository_name}/models/{version}/presign_download_url"
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_object(resp.json(), ModelSchema)

    def start_model_multipart_upload(
        self, model_repository_name: str, version: str
    ) -> ModelSchema:
        url = f"/api/v1/model_repositories/{model_repository_name}/models/{version}/start_multipart_upload"
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_object(resp.json(), ModelSchema)

    def presign_model_multipart_upload_url(
        self,
        model_repository_name: str,
        version: str,
        req: PreSignMultipartUploadUrlSchema,
    ) -> ModelSchema:
        url = f"/api/v1/model_repositories/{model_repository_name}/models/{version}/presign_multipart_upload_url"
        resp = self.session.patch(url, json=schema_to_object(req))
        self._check_resp(resp)
        return schema_from_object(resp.json(), ModelSchema)

    def complete_model_multipart_upload(
        self,
        model_repository_name: str,
        version: str,
        req: CompleteMultipartUploadSchema,
    ) -> ModelSchema:
        url = f"/api/v1/model_repositories/{model_repository_name}/models/{version}/complete_multipart_upload"
        resp = self.session.patch(url, json=schema_to_object(req))
        self._check_resp(resp)
        return schema_from_object(resp.json(), ModelSchema)

    def start_upload_model(
        self, model_repository_name: str, version: str
    ) -> ModelSchema:
        url = f"/api/v1/model_repositories/{model_repository_name}/models/{version}/start_upload"
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_object(resp.json(), ModelSchema)

    def finish_upload_model(
        self, model_repository_name: str, version: str, req: FinishUploadSchema
    ) -> ModelSchema:
        url = f"/api/v1/model_repositories/{model_repository_name}/models/{version}/finish_upload"
        resp = self.session.patch(url, json=schema_to_object(req))
        self._check_resp(resp)
        return schema_from_object(resp.json(), ModelSchema)

    def upload_model(
        self, model_repository_name: str, version: str, data: t.IO[bytes]
    ) -> None:
        url = f"/api/v1/model_repositories/{model_repository_name}/models/{version}/upload"
        resp = self.session.put(
            url,
            content=data,
            headers=dict(
                self.session.headers, **{"Content-Type": "application/octet-stream"}
            ),
        )
        self._check_resp(resp)
        return None

    @contextlib.contextmanager
    def download_model(
        self, model_repository_name: str, version: str
    ) -> t.Generator[httpx.Response, None, None]:
        url = f"/api/v1/model_repositories/{model_repository_name}/models/{version}/download"
        with self.session.stream("GET", url) as resp:
            self._check_resp(resp)
            yield resp

    def get_bento_repositories_list(
        self, bento_repository_name: str
    ) -> BentoWithRepositoryListSchema | None:
        url = "/api/v1/bento_repositories"
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoWithRepositoryListSchema)

    def get_bentos_list(self) -> BentoWithRepositoryListSchema | None:
        url = "/api/v1/bentos"
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), BentoWithRepositoryListSchema)

    def get_models_list(self) -> ModelWithRepositoryListSchema | None:
        url = "/api/v1/models"
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), ModelWithRepositoryListSchema)

    def get_cluster_deployment_list(
        self, cluster: str, **params: str | int | None
    ) -> DeploymentListSchema | None:
        url = f"/api/v1/clusters/{cluster}/deployments"
        resp = self.session.get(url, params=params)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentListSchema)

    def get_organization_deployment_list(
        self, **params: str | int | None
    ) -> DeploymentListSchema | None:
        url = "/api/v1/deployments"
        resp = self.session.get(url, params=params)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentListSchema)

    def create_deployment(
        self, cluster: str, create_schema: CreateDeploymentSchemaV1
    ) -> DeploymentFullSchema | None:
        url = f"/api/v1/clusters/{cluster}/deployments"
        resp = self.session.post(url, json=schema_to_object(create_schema))
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentFullSchema)

    def get_deployment(
        self, cluster: str, kube_namespace: str, name: str
    ) -> DeploymentFullSchema | None:
        url = (
            f"/api/v1/clusters/{cluster}/namespaces/{kube_namespace}/deployments/{name}"
        )
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentFullSchema)

    def update_deployment(
        self,
        cluster: str,
        kube_namespace: str,
        name: str,
        update_schema: UpdateDeploymentSchema,
    ) -> DeploymentFullSchema | None:
        url = (
            f"/api/v1/clusters/{cluster}/namespaces/{kube_namespace}/deployments/{name}"
        )
        resp = self.session.patch(url, json=schema_to_object(update_schema))
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentFullSchema)

    def terminate_deployment(
        self, cluster: str, kube_namespace: str, name: str
    ) -> DeploymentFullSchema | None:
        url = f"/api/v1/clusters/{cluster}/namespaces/{kube_namespace}/deployments/{name}/terminate"
        resp = self.session.post(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentFullSchema)

    def delete_deployment(
        self, cluster: str, kube_namespace: str, name: str
    ) -> DeploymentFullSchema | None:
        url = (
            f"/api/v1/clusters/{cluster}/namespaces/{kube_namespace}/deployments/{name}"
        )
        resp = self.session.delete(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentFullSchema)

    def get_cluster_list(
        self, params: dict[str, str | int] | None = None
    ) -> ClusterListSchema | None:
        url = "/api/v1/clusters"
        resp = self.session.get(url, params=params)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), ClusterListSchema)

    def get_cluster(self, cluster: str) -> ClusterFullSchema | None:
        url = f"/api/v1/clusters/{cluster}"
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_object(resp.json(), ClusterFullSchema)

    def get_latest_model(
        self, model_repository_name: str, query: str | None = None
    ) -> ModelSchema | None:
        url = f"/api/v1/model_repositories/{model_repository_name}/models"
        params = {"start": "0", "count": "10"}
        if query:
            params["q"] = query
        resp = self.session.get(url, params=params)
        self._check_resp(resp)
        models = resp.json()["items"]
        return schema_from_object(models[0], ModelSchema) if models else None

    def list_secrets(
        self,
        count: int | None = None,
        q: str | None = None,
        search: str | None = None,
        start: int | None = None,
    ) -> SecretListSchema:
        url = "/api/v1/org_secrets"
        if not count:
            count = 10
        if not start:
            start = 0
        resp = self.session.get(
            url,
            params={
                "count": count,
                "q": q,
                "search": search,
                "start": start,
                "all": "true",
            },
        )
        self._check_resp(resp)
        return schema_from_object(resp.json(), SecretListSchema)

    def create_secret(
        self, secret: CreateSecretSchema, cluster: str | None = None
    ) -> SecretSchema:
        url = "/api/v1/org_secrets"
        if cluster:
            url = f"{url}?cluster={cluster}"
        resp = self.session.post(url, json=schema_to_object(secret))
        self._check_resp(resp)
        return schema_from_object(resp.json(), SecretSchema)

    def get_secret(self, name: str, cluster: str | None = None) -> SecretSchema:
        url = f"/api/v1/org_secrets/{name}"
        if cluster:
            url = f"{url}?cluster={cluster}"
        resp = self.session.get(url)
        self._check_resp(resp)
        return schema_from_object(resp.json(), SecretSchema)

    def delete_secret(self, name: str, cluster: str | None = None) -> None:
        url = f"/api/v1/org_secrets/{name}"
        if cluster:
            url = f"{url}?cluster={cluster}"
        resp = self.session.delete(url)
        self._check_resp(resp)

    def update_secret(
        self, name: str, secret: UpdateSecretSchema, cluster: str | None = None
    ) -> SecretSchema:
        url = f"/api/v1/org_secrets/{name}"
        if cluster:
            url = f"{url}?cluster={cluster}"
        resp = self.session.patch(url, json=schema_to_object(secret))
        self._check_resp(resp)
        return schema_from_object(resp.json(), SecretSchema)


class RestApiClientV2(BaseRestApiClient):
    def create_deployment(
        self,
        create_schema: CreateDeploymentSchemaV2,
        cluster: str | None = None,
    ) -> DeploymentFullSchemaV2:
        url = "/api/v2/deployments"
        resp = self.session.post(
            url, json=schema_to_object(create_schema), params={"cluster": cluster}
        )
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentFullSchemaV2)

    def update_deployment(
        self,
        name: str,
        update_schema: UpdateDeploymentSchemaV2,
        cluster: str | None = None,
    ) -> DeploymentFullSchemaV2:
        url = f"/api/v2/deployments/{name}"
        resp = self.session.put(
            url, json=schema_to_object(update_schema), params={"cluster": cluster}
        )
        if self._is_not_found(resp):
            raise NotFound(f"Deployment {name} is not found: {resp.text}")
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentFullSchemaV2)

    def void_update_deployment(
        self, name: str, cluster: str | None
    ) -> DeploymentFullSchemaV2:
        url = f"/api/v2/deployments/{name}/void_update"
        resp = self.session.put(url, params={"cluster": cluster})
        if self._is_not_found(resp):
            raise NotFound(f"Deployment {name} is not found: {resp.text}")
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentFullSchemaV2)

    def get_deployment(
        self,
        name: str,
        cluster: str | None = None,
    ) -> DeploymentFullSchemaV2:
        url = f"/api/v2/deployments/{name}"
        resp = self.session.get(url, params={"cluster": cluster})
        if self._is_not_found(resp):
            raise NotFound(f"Deployment {name} is not found: {resp.text}")
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentFullSchemaV2)

    def list_deployment(
        self,
        cluster: str | None = None,
        all: bool | None = None,
        # if both of the above is none, list default cluster's deployments
        count: int | None = None,
        q: str | None = None,
        search: str | None = None,
        start: int | None = None,
    ) -> DeploymentListSchemaV2:
        url = "/api/v2/deployments"
        resp = self.session.get(
            url,
            params={
                "cluster": cluster,
                "all": all,
                "count": count,
                "q": q,
                "search": search,
                "start": start,
            },
        )
        if self._is_not_found(resp):
            raise NotFound(f"Deployment not found: {resp.text}")
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentListSchemaV2)

    def terminate_deployment(
        self,
        name: str,
        cluster: str | None = None,
    ) -> DeploymentFullSchemaV2:
        url = f"/api/v2/deployments/{name}/terminate"
        resp = self.session.post(url, params={"cluster": cluster})
        if self._is_not_found(resp):
            raise NotFound(f"Deployment {name} is not found: {resp.text}")
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentFullSchemaV2)

    def delete_deployment(
        self,
        name: str,
        cluster: str | None = None,
    ) -> DeploymentFullSchemaV2:
        url = f"/api/v2/deployments/{name}"
        resp = self.session.delete(url, params={"cluster": cluster})
        if self._is_not_found(resp):
            raise NotFound(f"Deployment {name} is not found: {resp.text}")
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentFullSchemaV2)

    def list_instance_types(
        self,
        cluster: str | None = None,
    ) -> list[ResourceInstanceSchema]:
        url = "/api/v1/instance_types"
        resp = self.session.get(url, params={"cluster": cluster})
        self._check_resp(resp)
        return schema_from_object(resp.json(), list[ResourceInstanceSchema])

    def get_deployment_image_builder_pod(
        self, name: str, cluster: str | None = None
    ) -> KubePodSchema | None:
        deployment = self.get_deployment(name, cluster=cluster)
        if not deployment.latest_revision:
            raise NotFound(f"Deployment {name} latest revision is not found")
        if not deployment.latest_revision.targets:
            raise NotFound(f"Deployment {name} latest revision targets is not found")
        target = deployment.latest_revision.targets[0]
        if not target:
            raise NotFound(f"Deployment {name} latest revision target is not found")
        if not target.bento:
            raise NotFound(
                f"Deployment {name} latest revision target bento is not found"
            )
        scheme = "wss"
        base_url = self.session.base_url
        if base_url.scheme == "http":
            scheme = "ws"
        endpoint = f"{scheme}://{base_url.netloc.decode('utf-8')}"
        with connect_ws(
            url=f"{endpoint}/ws/v1/clusters/{deployment.cluster.name}/pods?"
            f"{urlencode(dict(organization_name=deployment.cluster.organization_name, namespace=deployment.kube_namespace, selector=f'yatai.ai/bento-repository={target.bento.repository.name},yatai.ai/bento={target.bento.version}'))}",
            client=self.session,
        ) as ws:
            jsn = schema_from_object(ws.receive_json(), KubePodWSResponseSchema)
            if jsn.type == "error":
                raise CloudRESTApiClientError(jsn.message)
            for pod in jsn.payload or []:
                if pod.labels.get("yatai.ai/is-bento-image-builder") == "true":
                    return pod
        return None

    def list_deployment_pods(
        self, name: str, cluster: str | None = None
    ) -> list[KubePodSchema]:
        deployment = self.get_deployment(name, cluster=cluster)
        if not deployment.latest_revision:
            raise NotFound(f"Deployment {name} latest revision is not found")
        if not deployment.latest_revision.targets:
            raise NotFound(f"Deployment {name} latest revision targets is not found")
        target = deployment.latest_revision.targets[0]
        if not target:
            raise NotFound(f"Deployment {name} latest revision target is not found")
        if not target.bento:
            raise NotFound(
                f"Deployment {name} latest revision target bento is not found"
            )
        scheme = "wss"
        base_url = self.session.base_url
        if base_url.scheme == "http":
            scheme = "ws"
        endpoint = f"{scheme}://{base_url.netloc.decode('utf-8')}"
        with connect_ws(
            url=f"{endpoint}/ws/v2/deployments/{name}/pods?{urlencode(dict(organization_name=deployment.cluster.organization_name, cluster=deployment.cluster.name))}",
            client=self.session,
        ) as ws:
            jsn = schema_from_object(ws.receive_json(), KubePodWSResponseSchema)
            if jsn.type == "error":
                raise CloudRESTApiClientError(jsn.message)
            return jsn.payload or []

    def tail_logs(
        self,
        *,
        cluster_name: str,
        namespace: str,
        pod_name: str,
        container_name: str = "main",
        stop_event: threading.Event,
    ) -> t.Generator[str, None, None]:
        scheme = "wss"
        base_url = self.session.base_url
        if base_url.scheme == "http":
            scheme = "ws"
        endpoint = f"{scheme}://{base_url.netloc.decode('utf-8')}"

        with connect_ws(
            url=f"{endpoint}/ws/v1/clusters/{cluster_name}/tail?{urlencode(dict(namespace=namespace, pod_name=pod_name, timestamps='false'))}",
            client=self.session,
        ) as ws:
            req_id = str(uuid.uuid4())
            ws.send_json(
                {
                    "type": "data",
                    "payload": {
                        "id": req_id,
                        "container_name": container_name,
                        "follow": True,
                        "tail_lines": 50,
                    },
                }
            )

            def heartbeat() -> None:
                while True:
                    try:
                        ws.send_json({"type": "heartbeat"})
                    except (WebSocketNetworkError, LocalProtocolError):
                        pass
                    if stop_event.wait(5):
                        break

            heartbeat_thread = threading.Thread(target=heartbeat)
            heartbeat_thread.start()

            try:
                while True:
                    if stop_event.is_set():
                        break
                    try:
                        data = ws.receive_json(timeout=1)
                    except Empty:
                        continue
                    jsn = schema_from_object(data, LogWSResponseSchema)
                    if jsn.type == "error":
                        raise CloudRESTApiClientError(jsn.message or "Unknown error")
                    if jsn.type == "success" and jsn.message == "EOF":
                        break
                    if jsn.type == "heartbeat" or jsn.payload is None:
                        continue
                    for line in jsn.payload.items:
                        yield line
            except WebSocketNetworkError:
                pass
            finally:
                stop_event.set()
                heartbeat_thread.join()

    def upload_files(
        self, name: str, files: UploadDeploymentFilesSchema, cluster: str | None = None
    ) -> None:
        url = f"/api/v2/deployments/{name}/files"
        resp = self.session.post(
            url, json=schema_to_object(files), params={"cluster": cluster}, timeout=600
        )
        self._check_resp(resp)

    def delete_files(
        self, name: str, paths: DeleteDeploymentFilesSchema, cluster: str | None = None
    ) -> None:
        url = f"/api/v2/deployments/{name}/files"
        resp = self.session.request(
            "DELETE", url, json=schema_to_object(paths), params={"cluster": cluster}
        )
        self._check_resp(resp)

    def list_files(
        self, name: str, cluster: str | None = None
    ) -> DeploymentFileListSchema:
        url = f"/api/v2/deployments/{name}/files"
        resp = self.session.get(url, params={"cluster": cluster})
        self._check_resp(resp)
        return schema_from_object(resp.json(), DeploymentFileListSchema)


class RestApiClient:
    def __init__(self, endpoint: str, api_token: str, timeout: int = 60) -> None:
        self.endpoint = endpoint
        headers = {"X-YATAI-API-TOKEN": api_token, "X-Bentoml-Version": BENTOML_VERSION}
        self.session = httpx.Client(base_url=endpoint, timeout=timeout, headers=headers)
        self.v2 = RestApiClientV2(self.session)
        self.v1 = RestApiClientV1(self.session)
