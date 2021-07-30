from typing import Optional, List
from enum import Enum

from pydantic import BaseModel


class IoDescriptor(BaseModel):
    type: str
    options: Optional[dict] = None


class Api(BaseModel):
    name: str
    docs: Optional[str] = None
    input: IoDescriptor
    output: IoDescriptor
    http_options: Optional[dict] = None


class Model(BaseModel):
    name: str
    version: str
    type: str
    metadata: Optional[dict] = None
    options: Optional[dict] = None
    context: Optional[dict] = None


class Environment(BaseModel):
    requirements_txt: Optional[str] = None
    environment_yml: Optional[str] = None
    docker_base_image: Optional[str] = None


class Metadata(BaseModel):
    name: str
    version: str
    apis: Optional[List[Api]] = None
    models: Optional[List[Model]] = None
    labels: Optional[dict] = None
    readme: Optional[str] = None
    swagger_json: Optional[str] = None
    environment: Optional[Environment] = None


class StorageType(str, Enum):
    filesystem = 'filesystem'
    s3 = 's3'
    gcs = 'gcs'
    abs = 'abs'


class Uri(BaseModel):
    type: StorageType
    uri: str
    pre_signed_url: Optional[str] = None



class AddBundleRequest(BaseModel):
    metadata: Metadata


class AddBundleResponse(BaseModel):
    uri: Uri


class GetBundleResponse(BaseModel):
    metadata: Metadata
    uri: Uri

class ListBundleResponse(BaseModel):
    bundles: Optional[List[Metadata]] = None

