from bentoml.configuration.containers import BentoMLContainer


yatai_metrics_client = BentoMLContainer.yatai_metrics_client.get()

GRPC_SERVER_STARTED_COUNTER = yatai_metrics_client.Counter(
    "grpc_server_started_total",
    "Total number of RPCs started on the server.",
    ["grpc_type", "grpc_service", "grpc_method"],
)

GRPC_SERVER_STREAM_MSG_RECEIVED = yatai_metrics_client.Counter(
    "grpc_server_msg_received_total",
    "Total number of RPC stream messages received on the server.",
    ["grpc_type", "grpc_service", "grpc_method"],
)

GRPC_SERVER_STREAM_MSG_SENT = yatai_metrics_client.Counter(
    "grpc_server_msg_sent_total",
    "Total number of gRPC stream messages sent by the server.",
    ["grpc_type", "grpc_service", "grpc_method"],
)

GRPC_SERVER_HANDLED_HISTOGRAM = yatai_metrics_client.Histogram(
    "grpc_server_handling_seconds",
    "Histogram of response latency (seconds) of gRPC that had been application-level "
    "handled by the server.",
    ["grpc_type", "grpc_service", "grpc_method"],
)

GRPC_SERVER_HANDLED_TOTAL = yatai_metrics_client.Counter(
    "grpc_server_handled_total",
    "Total number of RPCs completed on the server, regardless of success or failure.",
    ["grpc_type", "grpc_service", "grpc_method", "grpc_code"],
)
