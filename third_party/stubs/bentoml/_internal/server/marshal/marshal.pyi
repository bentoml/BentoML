from aiohttp import BaseConnector as BaseConnector, ClientSession as ClientSession
from aiohttp.web import Application, Request as Request
from aiohttp_cors import ResourceOptions as ResourceOptions
from bentoml.configuration.containers import BentoMLContainer as BentoMLContainer
from bentoml.exceptions import RemoteException as RemoteException
from bentoml.marshal.dispatcher import CorkDispatcher as CorkDispatcher, NonBlockSema as NonBlockSema
from bentoml.marshal.utils import DataLoader as DataLoader, MARSHAL_REQUEST_HEADER as MARSHAL_REQUEST_HEADER
from bentoml.saved_bundle import load_bento_service_metadata as load_bento_service_metadata
from bentoml.saved_bundle.config import DEFAULT_MAX_BATCH_SIZE as DEFAULT_MAX_BATCH_SIZE, DEFAULT_MAX_LATENCY as DEFAULT_MAX_LATENCY
from bentoml.types import HTTPRequest as HTTPRequest, HTTPResponse as HTTPResponse
from typing import Any, Optional

logger: Any

def metrics_patch(cls): ...

class MarshalApp:
    outbound_unix_socket: Any
    outbound_host: Any
    outbound_port: Any
    outbound_workers: Any
    mb_max_batch_size: Any
    mb_max_latency: Any
    batch_handlers: Any
    max_request_size: Any
    tracer: Any
    enable_access_control: Any
    access_control_allow_origin: Any
    access_control_options: Any
    bento_service_metadata_pb: Any
    timeout: Any
    CONNECTION_LIMIT: Any
    def __init__(self, bento_bundle_path: str = ..., outbound_host: str = ..., outbound_port: int = ..., outbound_workers: int = ..., mb_max_batch_size: int = ..., mb_max_latency: int = ..., max_request_size: int = ..., outbound_unix_socket: str = ..., enable_access_control: bool = ..., access_control_allow_origin: Optional[str] = ..., access_control_options: Optional[ResourceOptions] = ..., timeout: int = ..., tracer=...) -> None: ...
    @property
    def cleanup_tasks(self): ...
    async def cleanup(self, _) -> None: ...
    def fetch_sema(self): ...
    def get_conn(self) -> BaseConnector: ...
    def get_client(self): ...
    def add_batch_handler(self, api_route, max_latency, max_batch_size) -> None: ...
    def setup_routes_from_pb(self, bento_service_metadata_pb) -> None: ...
    async def request_dispatcher(self, request: Request): ...
    async def relay_handler(self, request: Request): ...
    def get_app(self) -> Application: ...
    def run(self, port=...) -> None: ...
