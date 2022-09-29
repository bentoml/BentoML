import json
import typing as t
from enum import Enum
from typing import TYPE_CHECKING
from datetime import datetime

import attr
import cattr
from dateutil.parser import parse

time_format = "%Y-%m-%d %H:%M:%S.%f"


def datetime_encoder(time_obj: t.Optional[datetime]) -> t.Optional[str]:
    if not time_obj:
        return None
    return time_obj.strftime(time_format)


def datetime_decoder(datetime_str: t.Optional[str], _: t.Any) -> t.Optional[datetime]:
    if not datetime_str:
        return None
    return parse(datetime_str)


converter = cattr.Converter()

converter.register_unstructure_hook(datetime, datetime_encoder)
converter.register_structure_hook(datetime, datetime_decoder)


T = t.TypeVar("T")


def schema_from_json(json_content: str, cls: t.Type[T]) -> T:
    dct = json.loads(json_content)
    return converter.structure(dct, cls)


def schema_to_json(obj: t.Any) -> str:
    res = converter.unstructure(obj, obj.__class__)
    return json.dumps(res)


@attr.define
class BaseSchema:
    uid: str
    created_at: datetime
    updated_at: t.Optional[datetime]
    deleted_at: t.Optional[datetime]


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
    items: t.List[OrganizationSchema]


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
class BentoRunnerResourceSchema:
    cpu: t.Optional[t.Any]
    nvidia_gpu: t.Optional[t.Any]
    custom_resources: t.Optional[t.Any]


@attr.define
class BentoRunnerSchema:
    name: str
    runnable_type: t.Optional[str]
    models: t.Optional[t.List[str]]
    resource_config: t.Optional[BentoRunnerResourceSchema]


@attr.define
class BentoManifestSchema:
    service: str
    bentoml_version: str
    size_bytes: int
    apis: t.Dict[str, BentoApiSchema] = attr.field(factory=dict)
    models: t.List[str] = attr.field(factory=list)
    runners: t.Optional[t.List[BentoRunnerSchema]] = attr.field(factory=list)


if TYPE_CHECKING:
    TransmissionStrategy = t.Literal["presigned_url", "proxy"]
else:
    TransmissionStrategy = str


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

    transmission_strategy: t.Optional[TransmissionStrategy] = attr.field(default=None)
    upload_id: t.Optional[str] = attr.field(default=None)

    upload_started_at: t.Optional[datetime] = attr.field(default=None)
    upload_finished_at: t.Optional[datetime] = attr.field(default=None)
    build_at: datetime = attr.field(factory=datetime.now)


@attr.define
class BentoRepositorySchema(ResourceSchema):
    description: str
    latest_bento: t.Optional[BentoSchema]


@attr.define
class CreateBentoSchema:
    description: str
    version: str
    manifest: BentoManifestSchema
    build_at: datetime = attr.field(factory=datetime.now)
    labels: t.List[LabelItemSchema] = attr.field(factory=list)


@attr.define
class UpdateBentoSchema:
    manifest: t.Optional[BentoManifestSchema] = attr.field(default=None)
    labels: t.Optional[t.List[LabelItemSchema]] = attr.field(default=None)


@attr.define
class PreSignMultipartUploadUrlSchema:
    upload_id: str
    part_number: int


@attr.define
class CompletePartSchema:
    part_number: int
    etag: str


@attr.define
class CompleteMultipartUploadSchema:
    parts: t.List[CompletePartSchema]
    upload_id: str


@attr.define
class FinishUploadBentoSchema:
    status: t.Optional[BentoUploadStatus]
    reason: t.Optional[str]


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
    metadata: t.Dict[str, t.Any] = attr.field(factory=dict)
    context: t.Dict[str, t.Any] = attr.field(factory=dict)
    options: t.Dict[str, t.Any] = attr.field(factory=dict)


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

    transmission_strategy: t.Optional[TransmissionStrategy] = attr.field(default=None)
    upload_id: t.Optional[str] = attr.field(default=None)

    upload_started_at: t.Optional[datetime] = attr.field(default=None)
    upload_finished_at: t.Optional[datetime] = attr.field(default=None)
    build_at: datetime = attr.field(factory=datetime.now)


@attr.define
class ModelRepositorySchema(ResourceSchema):
    description: str
    latest_model: t.Optional[ModelSchema]


@attr.define
class CreateModelSchema:
    description: str
    version: str
    manifest: ModelManifestSchema
    build_at: datetime = attr.field(factory=datetime.now)
    labels: t.List[LabelItemSchema] = attr.field(factory=list)


@attr.define
class FinishUploadModelSchema:
    status: t.Optional[ModelUploadStatus]
    reason: t.Optional[str]
