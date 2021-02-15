# https://github.com/grpc-ecosystem/go-grpc-prometheus/blob/master/client_metrics.go
from prometheus_client import Counter, Histogram

# We can get total number of errors, start handled, response handled
GRPC_CLIENT_STARTED_COUNTER = Counter(
    "grpc_client_started_total",
    "Total number of RPC started from clients",
    ["grpc_type", "grpc_service", "grpc_method"],
)
GRPC_CLIENT_HANDLED_COUNTER = Counter(
    "grpc_client_handled_total",
    "Total number of RPC completed on clients",
    ["grpc_type", "grpc_service", "grpc_method", "grpc_code"],
)
GRPC_CLIENT_STREAM_MSG_RECEIVED = Counter(
    "grpc_client_msg_sent_total",
    "Total number of RPC streams messages received by the client",
    ["grpc_type", "grpc_service", "grpc_method"],
)
GRPC_CLIENT_STREAM_MSG_SENT = Counter(
    "grpc_client_stream_msg_sent",
    "Total number of gRPC stream messages sent by the client",
    ["grpc_type", "grpc_service", "grpc_method"],
)

GRPC_CLIENT_HANDLED_HISTORGRAM = Histogram(
    "grpc_client_handling_seconds",
    "Histogram of response latency (sec) of gRPC until it is finished by the application",
    ["grpc_type", "grpc_service", "grpc_method"],
)
GRPC_CLIENT_STREAM_RECV_HISTOGRAM = Histogram(
    "grpc_client_msg_recv_handling_seconds",
    "Histogram of response latency (seconds) of the gRPC single message receive.",
    ["grpc_type", "grpc_service", "grpc_method"],
)
GRPC_CLIENT_STREAM_SEND_HISTOGRAM = Histogram(
    "grpc_client_msg_send_handling_seconds",
    "Histogram of response latency (seconds) of the gRPC single message send.",
    ["grpc_type", "grpc_service", "grpc_method"],
)
