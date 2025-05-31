import asyncio
import time
import bentoml
from bentoml.io import Text # For any potential REST/HTTP endpoints, not used by gRPC directly

# Import generated gRPC stubs
from generated import example_service_pb2
from generated import example_service_pb2_grpc

# Servicer implementation
class SimpleStreamingServicerImpl(example_service_pb2_grpc.SimpleStreamingServiceServicer):
    async def Chat(self, request: example_service_pb2.ChatMessage, context):
        print(f"Received chat message from client: '{request.text}' (ID: {request.message_id})")
        
        for i in range(5): # Stream back 5 messages
            response_text = f"Response {i+1} to '{request.text}'"
            timestamp = int(time.time() * 1000) # Current timestamp in milliseconds
            message_id = f"server-msg-{timestamp}-{i}"
            
            print(f"Sending: '{response_text}' (ID: {message_id})")
            yield example_service_pb2.ChatMessage(
                message_id=message_id,
                text=response_text,
                timestamp=timestamp
            )
            await asyncio.sleep(0.5) # Simulate some work or delay
        print("Finished streaming responses for Chat.")

# Create a BentoML service
svc = bentoml.Service(
    name="custom_grpc_stream_example_service",
    runners={}, # No runners needed for this simple example
)

# Mount the gRPC servicer
# The gRPC server will be started by BentoML when using `bentoml serve-grpc` or `bentoml serve`
# (if gRPC is enabled in config or via CLI options).
simple_servicer = SimpleStreamingServicerImpl()
svc.mount_grpc_servicer(
    servicer_cls=SimpleStreamingServicerImpl, # The class of the servicer
    add_servicer_fn=example_service_pb2_grpc.add_SimpleStreamingServiceServicer_to_server, # The function to add it
    service_names=[example_service_pb2.DESCRIPTOR.services_by_name['SimpleStreamingService'].full_name] # Service names
)


# Example of a simple REST endpoint (optional, just to show it can coexist)
@svc.api(input=Text(), output=Text())
def greet(input_text: str) -> str:
    return f"Hello, {input_text}! This is the REST endpoint."

```
