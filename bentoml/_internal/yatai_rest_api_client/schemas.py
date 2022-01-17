import json
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import TypeVar
from typing import Optional
from datetime import datetime

import attr
import cattr
from dateutil.parser import parse

time_format = "%Y-%m-%d %H:%M:%S.%f"


def datetime_encoder(time_obj: Optional[datetime]) -> Optional[str]:
    if not time_obj:
        return None
    return time_obj.strftime(time_format)


def datetime_decoder(datetime_str: Optional[str], _) -> Optional[datetime]:
    if not datetime_str:
        return None
    return parse(datetime_str)


converter = cattr.Converter()

converter.register_unstructure_hook(datetime, datetime_encoder)
converter.register_structure_hook(datetime, datetime_decoder)


T = TypeVar("T")


def schema_from_json(json_content: str, cls: Type[T]) -> T:
    dct = json.loads(json_content)
    return converter.structure(dct, cls)


def schema_to_json(obj: T) -> str:
    res = converter.unstructure(obj, obj.__class__)
    return json.dumps(res)


@attr.define
class BaseSchema:
    uid: str
    created_at: datetime
    updated_at: Optional[datetime]
    deleted_at: Optional[datetime]


@attr.define
class BaseListSchema:
    start: int
    count: int
    total: int


class ResourceType(Enum):
    USER = "user"
    ORG = "organization"
    CLUSTER = "cluster"
    BENTO_REPOSITORY = "bento_repository"
    BENTO = "bento"
    MODEL_REPOSITORY = "model_repository"
    MODEL = "model"


@attr.define
class ResourceSchema(BaseSchema):
    name: str
    resource_type: ResourceType


@attr.define
class LabelItemSchema:
    key: str
    value: str


@attr.define
class UserSchema:
    name: str
    email: str
    first_name: str
    last_name: str

    def get_name(self) -> str:
        if not self.first_name and not self.last_name:
            return self.name
        return f"{self.first_name} {self.last_name}".strip()


@attr.define
class OrganizationSchema(ResourceSchema):
    description: str


@attr.define
class OrganizationListSchema(BaseListSchema):
    items: List[OrganizationSchema]


@attr.define
class ClusterSchema(ResourceSchema):
    description: str


@attr.define
class CreateBentoRepositorySchema:
    name: str
    description: str


class BentoImageBuildStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"
    SUCCESS = "success"
    FAILED = "failed"


class BentoUploadStatus(Enum):
    PENDING = "pending"
    BUILDING = "uploading"
    SUCCESS = "success"
    FAILED = "failed"


@attr.define
class BentoApiSchema:
    route: str
    doc: str
    input: str
    output: str


@attr.define
class BentoManifestSchema:
    service: str
    bentoml_version: str
    size_bytes: int
    apis: Dict[str, BentoApiSchema] = attr.field(factory=dict)
    models: List[str] = attr.field(factory=list)


@attr.define
class BentoSchema(ResourceSchema):
    description: str
    version: str
    image_build_status: BentoImageBuildStatus
    upload_status: BentoUploadStatus
    upload_finished_reason: str
    presigned_upload_url: str
    presigned_download_url: str
    manifest: BentoManifestSchema

    upload_started_at: Optional[datetime] = attr.field(default=None)
    upload_finished_at: Optional[datetime] = attr.field(default=None)
    build_at: datetime = attr.field(factory=datetime.now)


@attr.define
class BentoRepositorySchema(ResourceSchema):
    description: str
    latest_bento: Optional[BentoSchema]


@attr.define
class CreateBentoSchema:
    description: str
    version: str
    manifest: BentoManifestSchema
    build_at: datetime = attr.field(factory=datetime.now)
    labels: List[LabelItemSchema] = attr.field(factory=list)


@attr.define
class FinishUploadBentoSchema:
    status: Optional[BentoUploadStatus]
    reason: Optional[str]


@attr.define
class CreateModelRepositorySchema:
    name: str
    description: str


class ModelImageBuildStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"
    SUCCESS = "success"
    FAILED = "failed"


class ModelUploadStatus(Enum):
    PENDING = "pending"
    BUILDING = "uploading"
    SUCCESS = "success"
    FAILED = "failed"


@attr.define
class ModelManifestSchema:
    module: str
    api_version: str
    bentoml_version: str
    size_bytes: int
    metadata: Dict[str, Any] = attr.field(factory=dict)
    context: Dict[str, Any] = attr.field(factory=dict)
    options: Dict[str, Any] = attr.field(factory=dict)


@attr.define
class ModelSchema(ResourceSchema):
    description: str
    version: str
    image_build_status: ModelImageBuildStatus
    upload_status: ModelUploadStatus
    upload_finished_reason: str
    presigned_upload_url: str
    presigned_download_url: str
    manifest: ModelManifestSchema

    upload_started_at: Optional[datetime] = attr.field(default=None)
    upload_finished_at: Optional[datetime] = attr.field(default=None)
    build_at: datetime = attr.field(factory=datetime.now)


@attr.define
class ModelRepositorySchema(ResourceSchema):
    description: str
    latest_model: Optional[ModelSchema]


@attr.define
class CreateModelSchema:
    description: str
    version: str
    manifest: ModelManifestSchema
    build_at: datetime = attr.field(factory=datetime.now)
    labels: List[LabelItemSchema] = attr.field(factory=list)


@attr.define
class FinishUploadModelSchema:
    status: Optional[ModelUploadStatus]
    reason: Optional[str]
