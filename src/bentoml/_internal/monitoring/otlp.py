from __future__ import annotations

import collections
import datetime
import logging
import logging.config
import os
import random
import typing as t

# NOTE: AFAIK, they move this function from opentelemetry.sdk.logs to opentelemetry._logs
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs import LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_CERTIFICATE
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_COMPRESSION
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_ENDPOINT
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_HEADERS
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_INSECURE
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TIMEOUT
from opentelemetry.sdk.resources import Resource

from ...exceptions import MissingDependencyException
from ..context import server_context
from ..context import trace_context
from .base import MonitorBase

try:
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
        OTLPLogExporter as OTLPGrpcLogExporter,
    )
    from opentelemetry.exporter.otlp.proto.http._log_exporter import (
        OTLPLogExporter as OTLPHttpLogExporter,
    )
except ImportError:
    raise MissingDependencyException(
        "'opentelemetry-exporter-otlp' is required to use OTLP exporter. Make sure to install it with 'pip install \"bentoml[monitor-otlp]\""
    )

if t.TYPE_CHECKING:
    from ..types import JSONSerializable


class OTLPMonitor(MonitorBase["JSONSerializable"]):
    """
    The monitor implementation to log data to OTLP endpoint.
    The otlp exporter could be configured by environment variables or
    by passing arguments to the monitor from the config file.
    For more information, please refer to:
    * https://opentelemetry.io/docs/concepts/sdk-configuration/otlp-exporter-configuration/

    The sample output of the data is as follows:

    .. code-block:: json

        {
            "sepal length": 4.9,
            "sepal width": 3.0,
            "petal length": 1.4,
            "petal width": 0.2,
            "pred": "setosa",
            "timestamp": 1668660679.587727,
            "request_id": "6413930528999883196",
            "bento_meta": {
                "bento_name": "iris_classifier",
                "bento_version": "not available",
                "monitor_name": "iris_classifier_prediction",
                "schema": {
                    "sepal length": {
                        "name": "sepal length",
                        "role": "feature",
                        "type": "numerical",
                    },
                    "sepal width": {
                        "name": "sepal width",
                        "role": "feature",
                        "type": "numerical",
                    },
                    "petal length": {
                        "name": "petal length",
                        "role": "feature",
                        "type": "numerical",
                    },
                    "petal width": {
                        "name": "petal width",
                        "role": "feature",
                        "type": "numerical",
                    },
                    "pred": {"name": "pred", "role": "prediction", "type": "categorical"},
                },
            },
        }

    """

    PRESERVED_COLUMNS = (COLUMN_TIME, COLUMN_RID, COLUMN_TID, COLUMN_META) = (
        "timestamp",
        "request_id",
        "trace_id",
        "bento_meta",
    )

    def __init__(
        self,
        name: str,
        endpoint: str | None = None,
        insecure: bool | str | None = None,
        credentials: str | None = None,
        headers: str | None = None,
        timeout: int | str | None = None,
        compression: str | None = None,
        meta_sample_rate: float = 1.0,
        protocol: t.Literal["http", "grpc"] = "http",
        **_: t.Any,
    ) -> None:
        """
        Initialize the monitor.

        Args:
            name: The name of the monitor.
            endpoint: The endpoint to send the data to.
            insecure: Whether to use insecure connection.
            credentials: The credentials to use.
            headers: The headers to use.
            timeout: The timeout to use.
            compression: The compression to use.
            protocol: The protocol to use.
        """
        super().__init__(name, **_)

        self.endpoint = endpoint
        self.insecure = insecure
        self.credentials = credentials
        self.headers = headers
        self.timeout = timeout
        self.compression = compression
        self.meta_sample_rate = meta_sample_rate

        self.logger_provider: LoggerProvider | None = None
        self.data_logger: logging.Logger | None = None
        self._schema: dict[str, dict[str, str]] = {}
        self._will_export_schema = False

        self.protocol = protocol

    def _init_logger(self) -> None:
        from opentelemetry.sdk.resources import SERVICE_INSTANCE_ID
        from opentelemetry.sdk.resources import SERVICE_NAME
        from opentelemetry.sdk.resources import OTELResourceDetector

        # User can optionally configure the resource with the following environment variables. Only
        # configure resource if user has not explicitly configured it.
        system_otel_resources: Resource = OTELResourceDetector().detect()
        _resource = {}
        if server_context.bento_name:
            _resource[SERVICE_NAME] = f"{server_context.bento_name}:{self.name}"
        if server_context.bento_version:
            _resource[SERVICE_INSTANCE_ID] = server_context.bento_version

        bentoml_resource = Resource.create(_resource)

        self.logger_provider = LoggerProvider(
            resource=bentoml_resource.merge(system_otel_resources)
        )
        set_logger_provider(self.logger_provider)

        if self.endpoint is not None:
            os.environ[OTEL_EXPORTER_OTLP_ENDPOINT] = self.endpoint
        if self.insecure is not None:
            os.environ[OTEL_EXPORTER_OTLP_INSECURE] = str(self.insecure)
        if self.credentials is not None:
            os.environ[OTEL_EXPORTER_OTLP_CERTIFICATE] = self.credentials
        if self.headers is not None:
            os.environ[OTEL_EXPORTER_OTLP_HEADERS] = self.headers
        if self.timeout is not None:
            os.environ[OTEL_EXPORTER_OTLP_TIMEOUT] = str(self.timeout)
        if self.compression is not None:
            os.environ[OTEL_EXPORTER_OTLP_COMPRESSION] = self.compression

        exporter: OTLPHttpLogExporter | OTLPGrpcLogExporter
        if self.protocol == "http":
            exporter = OTLPHttpLogExporter(
                endpoint=self.endpoint,
                certificate_file=self.credentials,
                headers=self.headers,
                timeout=self.timeout,
                compression=self.compression,
            )
        elif self.protocol == "grpc":
            exporter = OTLPGrpcLogExporter(
                endpoint=self.endpoint,
                insecure=self.insecure,
                credentials=self.credentials,
                headers=self.headers,
                timeout=self.timeout,
                compression=self.compression,
            )
        else:
            raise ValueError(f"Invalid protocol: {self.protocol}")

        self.logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
        handler = LoggingHandler(
            level=logging.NOTSET, logger_provider=self.logger_provider
        )
        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "loggers": {
                    "bentoml_monitor_data": {
                        "level": "INFO",
                        "handlers": [],
                        "propagate": False,
                    }
                },
            }
        )
        self.data_logger = logging.getLogger("bentoml_monitor_data")
        self.data_logger.addHandler(handler)

    def __del__(self) -> None:
        if self.logger_provider is not None:
            self.logger_provider.shutdown()

    def export_schema(self, columns_schema: dict[str, dict[str, str]]) -> None:
        """
        Export columns_schema of the data. This method should be called after all data is logged.
        """
        self._schema = columns_schema
        self._will_export_schema = True

    def export_data(
        self,
        datas: t.Dict[str, collections.deque[JSONSerializable]],
    ) -> None:
        """
        Export data. This method should be called after all data is logged.
        """
        if self.data_logger is None:
            self._init_logger()
            assert self.data_logger is not None

        extra_columns: dict[str, t.Any] = {
            self.COLUMN_TIME: datetime.datetime.now().timestamp(),
            self.COLUMN_RID: str(trace_context.request_id),
            self.COLUMN_TID: str(trace_context.trace_id),
        }

        if self._will_export_schema or random.random() < self.meta_sample_rate:
            extra_columns[self.COLUMN_META] = {
                "bento_name": server_context.bento_name,
                "bento_version": server_context.bento_version,
                "monitor_name": self.name,
                "schema": self._schema,
            }

            self._will_export_schema = False

        while True:
            try:
                record = {k: v.popleft() for k, v in datas.items()}
                record.update(extra_columns)  # type: ignore  # pyright can not resolve backward reference correctly for JSONSerializable
                self.data_logger.info(record)
            except IndexError:
                break
