import logging
from typing import Optional
from urllib.parse import urljoin

import requests

from .schemas import UserSchema
from .schemas import BentoSchema
from .schemas import ModelSchema
from .schemas import schema_to_json
from .schemas import schema_from_json
from .schemas import CreateBentoSchema
from .schemas import CreateModelSchema
from .schemas import OrganizationSchema
from .schemas import BentoRepositorySchema
from .schemas import ModelRepositorySchema
from .schemas import FinishUploadBentoSchema
from .schemas import FinishUploadModelSchema
from .schemas import CreateBentoRepositorySchema
from .schemas import CreateModelRepositorySchema
from ...exceptions import YataiRESTApiClientError

logger = logging.getLogger(__name__)


class YataiRESTApiClient:
    def __init__(self, endpoint: str, api_token: str) -> None:
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-YATAI-API-TOKEN": api_token,
                "Content-Type": "application/json",
            }
        )

    def _is_not_found(self, resp: requests.Response) -> bool:
        # Forgive me, I don't know how to map the error returned by gorm to juju/errors
        return resp.status_code == 400 and "record not found" in resp.text

    def _check_resp(self, resp: requests.Response) -> None:
        if resp.status_code != 200:
            raise YataiRESTApiClientError(
                f"request failed with status code {resp.status_code}: {resp.text}"
            )

    def get_current_user(self) -> Optional[UserSchema]:
        url = urljoin(self.endpoint, "/api/v1/auth/current")
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_json(resp.text, UserSchema)

    def get_current_organization(self) -> Optional[OrganizationSchema]:
        url = urljoin(self.endpoint, "/api/v1/current_org")
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_json(resp.text, OrganizationSchema)

    def get_bento_repository(
        self, bento_repository_name: str
    ) -> Optional[BentoRepositorySchema]:
        url = urljoin(
            self.endpoint, f"/api/v1/bento_repositories/{bento_repository_name}"
        )
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_json(resp.text, BentoRepositorySchema)

    def create_bento_repository(
        self, req: CreateBentoRepositorySchema
    ) -> BentoRepositorySchema:
        url = urljoin(self.endpoint, "/api/v1/bento_repositories")
        resp = self.session.post(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, BentoRepositorySchema)

    def get_bento(
        self, bento_repository_name: str, version: str
    ) -> Optional[BentoSchema]:
        url = urljoin(
            self.endpoint,
            f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}",
        )
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_json(resp.text, BentoSchema)

    def create_bento(
        self, bento_repository_name: str, req: CreateBentoSchema
    ) -> BentoSchema:
        url = urljoin(
            self.endpoint, f"/api/v1/bento_repositories/{bento_repository_name}/bentos"
        )
        resp = self.session.post(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, BentoSchema)

    def presign_bento_upload_url(
        self, bento_repository_name: str, version: str
    ) -> BentoSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}/presign_upload_url",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, BentoSchema)

    def presign_bento_download_url(
        self, bento_repository_name: str, version: str
    ) -> BentoSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}/presign_download_url",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, BentoSchema)

    def start_upload_bento(
        self, bento_repository_name: str, version: str
    ) -> BentoSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}/start_upload",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, BentoSchema)

    def finish_upload_bento(
        self, bento_repository_name: str, version: str, req: FinishUploadBentoSchema
    ) -> BentoSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/bento_repositories/{bento_repository_name}/bentos/{version}/finish_upload",
        )
        resp = self.session.patch(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, BentoSchema)

    def get_model_repository(
        self, model_repository_name: str
    ) -> Optional[ModelRepositorySchema]:
        url = urljoin(
            self.endpoint, f"/api/v1/model_repositories/{model_repository_name}"
        )
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelRepositorySchema)

    def create_model_repository(
        self, req: CreateModelRepositorySchema
    ) -> ModelRepositorySchema:
        url = urljoin(self.endpoint, "/api/v1/model_repositories")
        resp = self.session.post(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelRepositorySchema)

    def get_model(
        self, model_repository_name: str, version: str
    ) -> Optional[ModelSchema]:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}",
        )
        resp = self.session.get(url)
        if self._is_not_found(resp):
            return None
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def create_model(
        self, model_repository_name: str, req: CreateModelSchema
    ) -> ModelSchema:
        url = urljoin(
            self.endpoint, f"/api/v1/model_repositories/{model_repository_name}/models"
        )
        resp = self.session.post(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def presign_model_upload_url(
        self, model_repository_name: str, version: str
    ) -> ModelSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}/presign_upload_url",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def presign_model_download_url(
        self, model_repository_name: str, version: str
    ) -> ModelSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}/presign_download_url",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def start_upload_model(
        self, model_repository_name: str, version: str
    ) -> ModelSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}/start_upload",
        )
        resp = self.session.patch(url)
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)

    def finish_upload_model(
        self, model_repository_name: str, version: str, req: FinishUploadModelSchema
    ) -> ModelSchema:
        url = urljoin(
            self.endpoint,
            f"/api/v1/model_repositories/{model_repository_name}/models/{version}/finish_upload",
        )
        resp = self.session.patch(url, data=schema_to_json(req))
        self._check_resp(resp)
        return schema_from_json(resp.text, ModelSchema)
