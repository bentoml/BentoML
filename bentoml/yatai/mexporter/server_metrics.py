from prometheus_client import Counter, Historgram

GRPC_SERVER_STARTED_TOTAL_COUNTER = Counter(
    "grpc_server_started_total",
    "Total number of RPCs started on the server.",
    ["grpc_type", "grpc_service", "grpc_method"],
)

GRPC_SERVER_HANDLED_TOTAL_COUNTER = Counter(
    "grpc_server_handled_total",
    "Total number of RPCs completed on the server, regardless of success or failure.",
    ["grpc_type", "grpc_service", "grpc_method", "code"],
)

GRPC_SERVER_HANDLED_LATENCY_SECONDS = Histogram(
    "grpc_server_handled_latency_seconds",
    "Histogram of response latency (seconds) of gRPC that had been "
    "application-level handled by the server",
    ["grpc_type", "grpc_service", "grpc_method"],
)

GRPC_SERVER_MSG_RECEIVED_TOTAL_COUNTER = Counter(
    "grpc_server_msg_received_total",
    "Histogram of response latency (seconds) of gRPC that had been application-level "
    "handled by the server.",
    ["grpc_type", "grpc_service", "grpc_method"],
)

GRPC_SERVER_MSG_SENT_TOTAL_COUNTER = Counter(
    "grpc_server_msg_sent_total",
    "Total number of stream messages sent by the server.",
    ["grpc_type", "grpc_service", "grpc_method"],
)
