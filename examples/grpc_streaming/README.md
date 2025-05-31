# BentoML Custom gRPC Streaming Example

This example demonstrates how to define, implement, and serve a custom gRPC streaming service with BentoML. The service implements a simple "chat" style interaction where the client sends a message and the server streams back a series of responses.

## Files

- `protos/example_service.proto`: Protocol buffer definition for the `SimpleStreamingService`.
- `service.py`: BentoML service implementation that includes the gRPC servicer for `SimpleStreamingService`.
- `bentofile.yaml`: BentoML build configuration file.
- `client_example.py`: A Python script demonstrating how to call the gRPC streaming service.
- `generated/`: Directory containing Python stubs generated from `example_service.proto`.

## Prerequisites

- Python 3.8+
- BentoML installed (`pip install bentoml`)
- gRPC tools (`pip install grpcio grpcio-tools`)

## Setup

1.  **Generate gRPC Stubs**:
    Navigate to the `examples/grpc_streaming` directory and run:
    ```bash
    mkdir generated
    python -m grpc_tools.protoc -Iprotos --python_out=generated --grpc_python_out=generated protos/example_service.proto
    # Create __init__.py files to make them importable
    touch generated/__init__.py
    ```
    This will generate `example_service_pb2.py` and `example_service_pb2_grpc.py` in the `generated` directory.

## Running the Example

1.  **Serve the BentoML Service**:
    From the `examples/grpc_streaming` directory:
    ```bash
    bentoml serve service:svc --reload
    ```
    This will start the BentoML server, which by default includes a gRPC server on a port like `50051` (or the next available one, check console output, typically it is `[::]:<port specified in bentoml_config_options.yml or default>`). For this example, we assume the gRPC server runs on the default port that `bentoml serve` would expose if not configured, which might require checking BentoML's default configuration or explicitly setting it.
    *Note: `bentoml serve` starts HTTP server by default. For gRPC, you usually use `bentoml serve-grpc`. However, BentoML services can expose both. We will use `mount_grpc_servicer` which should make it available via the main gRPC server that `serve-grpc` would typically manage.*

    To ensure it uses a known gRPC port (e.g., 50051 if not default for `serve`), you might run:
    ```bash
    bentoml serve service:svc --reload --grpc-port 50051 
    # Or more explicitly for gRPC focus:
    # bentoml serve-grpc service:svc --reload --port 50051
    ```
    Check the output from `bentoml serve` for the actual gRPC port if you don't specify one. For this example, `client_example.py` assumes `localhost:50051`.

2.  **Run the Client**:
    In a new terminal, from the `examples/grpc_streaming` directory:
    ```bash
    python client_example.py
    ```

## Expected Output (Client)

```
Client sending: Hello, stream!
Server says: Response 1 to 'Hello, stream!'
Server says: Response 2 to 'Hello, stream!'
Server says: Response 3 to 'Hello, stream!'
Server says: Response 4 to 'Hello, stream!'
Server says: Response 5 to 'Hello, stream!'
Stream finished.
```

## How it Works

-   **`example_service.proto`**: Defines a `SimpleStreamingService` with a server-streaming RPC method `Chat`.
-   **`service.py`**:
    -   Implements `SimpleStreamingServicerImpl` which provides the logic for the `Chat` method.
    -   Creates a BentoML `Service` named `custom_grpc_stream_example`.
    -   Mounts the `SimpleStreamingServicerImpl` to the BentoML service instance. When this BentoML service is run with gRPC enabled, the custom gRPC service will be available.
-   **`client_example.py`**:
    -   Uses `grpc.insecure_channel` to connect to the server.
    -   Creates a stub for `SimpleStreamingService`.
    -   Calls the `Chat` method and iterates over the streamed responses.

This example showcases how to integrate custom gRPC services with streaming capabilities within the BentoML framework.
```
