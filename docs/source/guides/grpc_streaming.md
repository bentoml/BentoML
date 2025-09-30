# gRPC Streaming with BentoML (v1alpha1)

BentoML supports gRPC streaming, allowing for efficient, long-lived communication channels between clients and servers. This guide demonstrates how to define, implement, and use gRPC streaming services with BentoML's `v1alpha1` gRPC protocol.

This `v1alpha1` protocol is an initial version focused on bi-directional streaming where the client sends a single message and the server responds with a stream of messages.

## 1. Defining the Service (.proto)

First, define your service and messages using Protocol Buffers. For the `v1alpha1` streaming interface, BentoML provides a specific service definition. If you were building custom services beyond the default `BentoService`, you'd create your own `.proto` similar to this.

The core `v1alpha1` service used internally by BentoML is defined in `src/bentoml/grpc/v1alpha1/bentoml_service_v1alpha1.proto`:

```protobuf
syntax = "proto3";

package bentoml.grpc.v1alpha1;

// The BentoService service definition.
service BentoService {
  // A streaming RPC method that accepts a Request message
  // and returns a stream of Response messages.
  rpc CallStream(Request) returns (stream Response) {}
}

// The request message containing the input data.
message Request {
  string data = 1;
}

// The response message containing the output data.
message Response {
  string data = 1;
}
```

Key aspects:
- `service BentoService`: Defines the service name.
- `rpc CallStream(Request) returns (stream Response) {}`: This declares a server-streaming RPC method. The client sends a single `Request`, and the server replies with a stream of `Response` messages.

After defining your `.proto` file, you need to generate the Python gRPC stubs:
```bash
pip install grpcio-tools
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. your_service.proto
```
For BentoML's internal `v1alpha1` service, these stubs (`bentoml_service_v1alpha1_pb2.py` and `bentoml_service_v1alpha1_pb2_grpc.py`) are already generated and included.

## 2. Implementing the Server-Side Streaming Logic

You implement the server-side logic by creating a class that inherits from the generated `YourServiceServicer` (e.g., `BentoServiceServicer` for the internal service) and overriding the streaming methods.

Here's how the internal `BentoServiceImpl` for `v1alpha1` is structured (simplified from `src/bentoml/grpc/v1alpha1/server.py`):

```python
import asyncio
import grpc
# Assuming stubs are generated in 'generated' directory or available in path
from bentoml.grpc.v1alpha1 import bentoml_service_v1alpha1_pb2 as pb
from bentoml.grpc.v1alpha1 import bentoml_service_v1alpha1_pb2_grpc as services

class BentoServiceImpl(services.BentoServiceServicer):
    async def CallStream(self, request: pb.Request, context: grpc.aio.ServicerContext) -> pb.Response:
        """
        Example CallStream implementation.
        Receives a Request and yields a stream of Response messages.
        """
        print(f"CallStream received: {request.data}")
        for i in range(5): # Example: send 5 messages
            response_data = f"Response {i+1} for '{request.data}'"
            print(f"Sending: {response_data}")
            await asyncio.sleep(0.5) # Simulate work
            yield pb.Response(data=response_data)
        print("CallStream finished.")

# To run this service (example standalone server):
async def run_server(port=50051):
    server = grpc.aio.server()
    services.add_BentoServiceServicer_to_server(BentoServiceImpl(), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f"gRPC server started on port {port}")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(run_server())
```

When integrating with `bentoml serve-grpc`, BentoML handles running the gRPC server. You need to ensure your service implementation is correctly picked up, which is done by modifying `Service.get_grpc_servicer` if you are customizing the main BentoService, or by mounting your own servicer for custom services. For the `v1alpha1` protocol, BentoML's `Service` class is already configured to use this `BentoServiceImpl`.

## 3. Using the BentoMlGrpcClient (v1alpha1)

BentoML provides a client SDK to interact with the `v1alpha1` gRPC streaming service.

Example usage (from `src/bentoml/grpc/v1alpha1/client.py`):
```python
import asyncio
from bentoml.grpc.v1alpha1.client import BentoMlGrpcClient

async def main():
    client = BentoMlGrpcClient(host="localhost", port=50051)

    input_data = "Hello Streaming World"
    print(f"Calling CallStream with data: '{input_data}'")

    try:
        idx = 0
        async for response in client.call_stream(data=input_data):
            print(f"Received from stream (message {idx}): {response.data}")
            idx += 1
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await client.close()
        print("Client connection closed.")

if __name__ == "__main__":
    asyncio.run(main())
```
The `client.call_stream(data=...)` method returns an asynchronous iterator that yields `Response` messages from the server.

## 4. Using the `call-grpc-stream` CLI Command

BentoML provides a CLI command to easily test and interact with `v1alpha1` gRPC streaming services.

**Command Syntax:**
```bash
bentoml call-grpc-stream --host <hostname> --port <port_number> --data "<your_request_data>"
```

**Example:**
Assuming your BentoML gRPC server (with `v1alpha1` protocol) is running on `localhost:50051`:
```bash
bentoml call-grpc-stream --host localhost --port 50051 --data "Test Message from CLI"
```

Output will be similar to:
```
Connecting to gRPC server at localhost:50051...
Sending data: 'Test Message from CLI' to CallStream...
--- Streamed Responses ---
Response 1 for 'Test Message from CLI'
Response 2 for 'Test Message from CLI'
Response 3 for 'Test Message from CLI'
... (based on server implementation) ...
------------------------
Connection closed.
```

This CLI command uses the `BentoMlGrpcClient` internally.

## Summary

The `v1alpha1` gRPC streaming support in BentoML provides a foundation for building services that require persistent, streamed communication. By defining services in `.proto` files, implementing the server-side logic, and using the provided client SDK or CLI, you can leverage gRPC streaming in your BentoML applications. Remember that this `v1alpha1` version is specific to a client-sends-one, server-streams-many interaction pattern for the main `BentoService`. For more complex gRPC patterns (client-streaming, bidirectional-streaming for custom services), you would define those in your own `.proto` files and implement corresponding servicers and clients.
```
