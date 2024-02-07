from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Union

from annotated_types import Ge
from annotated_types import Gt
from annotated_types import Le
from pydantic import IPvAnyAddress
from pydantic import TypeAdapter
from typing_extensions import Annotated
from typing_extensions import Literal
from typing_extensions import TypedDict

Posint = Annotated[int, Gt(0)]
Posfloat = Annotated[float, Gt(0.0)]


class TrafficSchema(TypedDict, total=False):
    timeout: Posfloat
    max_concurrency: Posint
    external_queue: bool  # bentocloud only


class ResourceSchema(TypedDict, total=False):
    cpu: str
    memory: str
    gpu: Posfloat
    """
    gpu type defined here is only a anotation, it will use as an recommendation choice of instance type when deploying this service to bentocloud
    gpu_type follows the naming convention of AWS EC2 GPU instances, GCP GPU instances etc.
    """
    gpu_type: Literal[
        "nvidia-tesla-t4",
        "nvidia-tesla-a100",
        "nvidia-a100-80gb",
        "nvidia-a10g",
        "nvidia-l4",
        "nvidia-tesla-v100",
        "nvidia-tesla-p100",
        "nvidia-tesla-k80",
        "nvidia-tesla-p4",
    ]
    tpu_type: Literal[
        "v4-2x2x1",
        "v4-2x2x2",
        "v4-2x2x4",
        "v4-2x4x4",
        "v5p-2x2x1",
        "v5p-2x2x2",
        "v5p-2x2x4",
        "v5p-2x4x4",
        "v5e-1x1",
        "v5e-2x2",
        "v5e-2x4",
        "v5e-4x4",
        "v5e-4x8",
    ]


WorkerSchema = Union[Posint, Literal["cpu_count"]]


class MetricDuration(TypedDict, total=False):
    buckets: List[float]
    min: Annotated[float, Gt(0)]
    max: Annotated[float, Gt(0)]
    factor: Annotated[float, Gt(1.0)]


class MetricSchema(TypedDict, total=False):
    enabled: bool
    namespace: str
    duration: MetricDuration


class AccessLoggingSchema(TypedDict, total=False):
    enabled: bool
    skip_paths: List[str]
    request_content_length: bool
    request_content_type: bool
    response_content_length: bool
    response_content_type: bool
    format: TypedDict(
        "AccessLoggingFormat", {"trace_id": str, "span_id": str}, total=False
    )  # type: ignore


class SSLSchema(TypedDict, total=False):
    enabled: bool
    certfile: str
    keyfile: str
    keyfile_password: str
    ca_certs: str
    version: Posint
    cert_reqs: int
    ciphers: str


class HTTPCorsSchema(TypedDict, total=False):
    enabled: bool
    access_control_allow_origins: Union[str, List[str]]
    access_control_allow_credentials: bool
    access_control_allow_methods: Union[str, List[str]]
    access_control_allow_headers: Union[str, List[str]]
    access_control_allow_origin_regex: str
    access_control_max_age: int
    access_control_expose_headers: Union[str, List[str]]


class HTTPSchema(TypedDict, total=False):
    host: IPvAnyAddress
    port: int
    cors: HTTPCorsSchema
    response: TypedDict("HTTPResponseSchema", {"trace_id": bool}, total=False)  # type: ignore


class HostPortSchema(TypedDict):
    host: IPvAnyAddress
    port: Posint


class EnabledSchema(TypedDict):
    enabled: bool


class GRPCSchema(TypedDict, total=False):
    host: IPvAnyAddress
    port: Posint
    max_concurrent_streams: int
    maximum_concurrent_rpcs: int
    metrics: HostPortSchema
    reflection: EnabledSchema
    channelz: EnabledSchema
    max_message_length: int


class RunnerProbeSchema(TypedDict, total=False):
    enabled: bool
    timeout: Posint
    period: Posint


class MonitoringSchema(TypedDict, total=False):
    enabled: bool
    type: str
    options: Dict[str, Any]


class ZipkinSchema(TypedDict, total=False):
    endpoint: str
    local_node_ipv4: Union[IPvAnyAddress, int]
    local_node_ipv6: Union[IPvAnyAddress, int]
    local_node_port: Posint


class JaegerSchema(TypedDict, total=False):
    protocol: Literal["grpc", "thrift"]
    collector_endpoint: str
    grpc: TypedDict("JaegerGRPCSchema", {"insecure": bool}, total=False)  # type: ignore
    thrift: TypedDict(
        "JaegerThriftSchema",
        {
            "agent_host_name": str,
            "agent_port": Posint,
            "udp_split_oversized_batches": bool,
        },
        total=False,
    )  # type: ignore


class OTLPSchema(TypedDict, total=False):
    protocol: Literal["grpc", "http"]
    endpoint: str
    compression: Literal["gzip", "none", "deflate"]
    http: TypedDict(
        "OTLPHTTPSchema",
        {"headers": Dict[str, str], "certificate_file": str},
        total=False,
    )  # type: ignore
    grpc: TypedDict(
        "OTLPGRPCSchema",
        {"insecure": bool, "headers": Sequence[Sequence[str]]},
        total=False,
    )  # type: ignore


class TracingSchema(TypedDict, total=False):
    exporter_type: Literal["zipkin", "jaeger", "otlp", "in_memory"]
    sample_rate: Annotated[float, Ge(0.0), Le(1.0)]
    timeout: Posint
    max_tag_value_length: Posint
    excluded_urls: Union[str, List[str]]
    zipkin: ZipkinSchema
    jaeger: JaegerSchema
    otlp: OTLPSchema


class LoggingSchema(TypedDict, total=False):
    access: AccessLoggingSchema


class ServiceConfig(TypedDict, total=False):
    name: str
    traffic: TrafficSchema
    backlog: Annotated[int, Ge(64)]
    max_runner_connections: Posint
    resources: ResourceSchema
    workers: WorkerSchema
    threads: Posint
    metrics: MetricSchema
    logging: LoggingSchema
    ssl: SSLSchema
    http: HTTPSchema
    grpc: GRPCSchema
    runner_probe: RunnerProbeSchema
    tracing: TracingSchema
    monitoring: MonitoringSchema


schema_type = TypeAdapter(ServiceConfig)


def validate(data: ServiceConfig) -> ServiceConfig:
    return schema_type.validate_python(data)
