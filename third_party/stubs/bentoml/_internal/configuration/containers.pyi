from ..configuration import expand_env_var as expand_env_var, get_bentoml_deploy_version as get_bentoml_deploy_version
from ..exceptions import BentoMLConfigException as BentoMLConfigException
from ..server.marshal.marshal import MarshalApp as MarshalApp
from ..utils import get_free_port as get_free_port
from ..utils.ruamel_yaml import YAML as YAML
from simple_di import Provider as Provider
from typing import Any

LOGGER: Any
YATAI_REPOSITORY_S3: str
YATAI_REPOSITORY_GCS: str
YATAI_REPOSITORY_FILE_SYSTEM: str
YATAI_REPOSITORY_TYPES: Any
SCHEMA: Any

class BentoMLConfiguration:
    config: Any
    def __init__(self, default_config_file: str = ..., override_config_file: str = ..., validate_schema: bool = ...) -> None: ...
    def override(self, keys: list, value): ...
    def as_dict(self) -> dict: ...

class BentoMLContainerClass:
    config: Any
    @staticmethod
    def tracer(tracer_type: str = ..., zipkin_server_url: str = ..., jaeger_server_address: str = ..., jaeger_server_port: int = ...): ...
    @staticmethod
    def access_control_options(allow_credentials=..., expose_headers=..., allow_methods=..., allow_headers=..., max_age=...): ...
    api_server_workers: Any
    bentoml_home: Any
    bundle_path: Provider[str]
    service_host: Provider[str]
    service_port: Provider[int]
    forward_host: Provider[str]
    forward_port: Provider[int]
    @staticmethod
    def model_server(): ...
    @staticmethod
    def proxy_server(): ...
    @staticmethod
    def proxy_app() -> MarshalApp: ...
    @staticmethod
    def model_app(): ...
    prometheus_lock: Any
    prometheus_multiproc_dir: Any
    @staticmethod
    def metrics_client(multiproc_lock=..., multiproc_dir=..., namespace=...): ...
    @staticmethod
    def yatai_metrics_client(): ...
    bento_bundle_deployment_version: Any
    yatai_database_url: Any
    yatai_file_system_directory: Any
    yatai_tls_root_ca_cert: Any
    logging_file_directory: Any
    yatai_logging_path: Any

BentoMLContainer: Any
