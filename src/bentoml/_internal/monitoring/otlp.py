from __future__ import annotations

import os
import typing as t
import logging
import datetime
import collections
import logging.config
from typing import TYPE_CHECKING

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs import LoggingHandler
from opentelemetry.sdk._logs import set_logger_provider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_HEADERS
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_TIMEOUT
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_ENDPOINT
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_INSECURE
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_CERTIFICATE
from opentelemetry.sdk.environment_variables import OTEL_EXPORTER_OTLP_COMPRESSION
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

from .api import MonitorBase
from ..context import trace_context
from ..context import component_context

if TYPE_CHECKING:
    JSONSerializable = t.Union[
        str,
        int,
        float,
        bool,
        None,
        t.List["JSONSerializable"],
        t.Dict[str, "JSONSerializable"],
    ]


class OTLPMonitor(MonitorBase["JSONSerializable"]):
    """
    The monitor implementation to log data to OTLP endpoint.
    The otlp exporter could be configured by environment variables or
    by passing arguments to the monitor from the config file.

    For more information, please refer to:
    * https://opentelemetry.io/docs/concepts/sdk-configuration/otlp-exporter-configuration/
    """

    PRESERVED_COLUMNS = (COLUMN_TIME, COLUMN_RID, COLUMN_META, COLUMN_SCHEMA) = (
        "timestamp",
        "request_id",
        "bento_meta",
        "schema",
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
        """
        super().__init__(name, **_)

        self.endpoint = endpoint
        self.insecure = insecure
        self.credentials = credentials
        self.headers = headers
        self.timeout = timeout
        self.compression = compression

        self.logger_provider: LoggerProvider | None = None
        self.data_logger: logging.Logger | None = None
        self._schema: dict[str, dict[str, str]] = {}

    def _init_logger(self) -> None:
        self.logger_provider = LoggerProvider(
            resource=Resource.create(
                {
                    "service.name": f"{component_context.bento_name}:{self.name}",
                    "service.instance.id": "{component_context.bento_version}",
                }
            ),
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

        exporter = OTLPLogExporter()
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
                        "handlers": [handler],
                        "propagate": False,
                    }
                },
            }
        )
        self.data_logger = logging.getLogger("bentoml_monitor_schema")

    def __del__(self) -> None:
        if self.logger_provider is not None:
            self.logger_provider.shutdown()

    def export_schema(self, columns_schema: dict[str, dict[str, str]]) -> None:
        """
        Export columns_schema of the data. This method should be called after all data is logged.
        """
        self._schema = columns_schema

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

        extra_columns = {
            self.COLUMN_TIME: datetime.datetime.now().timestamp(),
            self.COLUMN_RID: str(trace_context.request_id),
            self.COLUMN_META: {
                "bento_name": component_context.bento_name,
                "bento_version": component_context.bento_version,
                "monitor_name": self.name,
            },
            self.COLUMN_SCHEMA: self._schema,
        }
        while True:
            try:
                record = {k: v.popleft() for k, v in datas.items()}
                record.update(extra_columns)  # type: ignore
                self.data_logger.info(record)
            except IndexError:
                break
