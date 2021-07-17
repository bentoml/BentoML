from typing import Any
from yatai.yatai.db import DB as DB
from yatai.yatai.db.stores.lock import LockStore as LockStore
from yatai.yatai.deployment.docker_utils import ensure_docker_available_or_raise as ensure_docker_available_or_raise
from yatai.yatai.deployment.operator import get_deployment_operator as get_deployment_operator
from yatai.yatai.grpc_stream_utils import DownloadBentoStreamResponses as DownloadBentoStreamResponses
from yatai.yatai.locking.lock import DEFAULT_TTL_MIN as DEFAULT_TTL_MIN, LockType as LockType, lock as lock
from yatai.yatai.proto import status_pb2 as status_pb2
from yatai.yatai.proto.deployment_pb2 import ApplyDeploymentResponse as ApplyDeploymentResponse, DeleteDeploymentResponse as DeleteDeploymentResponse, DeploymentSpec as DeploymentSpec, DescribeDeploymentResponse as DescribeDeploymentResponse, GetDeploymentResponse as GetDeploymentResponse, ListDeploymentsResponse as ListDeploymentsResponse
from yatai.yatai.proto.repository_pb2 import AddBentoResponse as AddBentoResponse, BentoUri as BentoUri, ContainerizeBentoResponse as ContainerizeBentoResponse, DangerouslyDeleteBentoResponse as DangerouslyDeleteBentoResponse, DownloadBentoResponse as DownloadBentoResponse, GetBentoResponse as GetBentoResponse, ListBentoResponse as ListBentoResponse, UpdateBentoResponse as UpdateBentoResponse, UploadBentoResponse as UploadBentoResponse, UploadStatus as UploadStatus
from yatai.yatai.proto.yatai_service_pb2 import GetYataiServiceVersionResponse as GetYataiServiceVersionResponse, HealthCheckResponse as HealthCheckResponse
from yatai.yatai.repository.base_repository import BaseRepository as BaseRepository
from yatai.yatai.repository.file_system_repository import FileSystemRepository as FileSystemRepository
from yatai.yatai.status import Status as Status
from yatai.yatai.utils import docker_build_logs as docker_build_logs
from yatai.yatai.validator import validate_deployment_pb as validate_deployment_pb

logger: Any

def track_deployment_delete(deployment_operator, created_at, force_delete: bool = ...) -> None: ...
def is_file_system_repo(repo_instance) -> bool: ...
def get_yatai_service_impl(base=...): ...
