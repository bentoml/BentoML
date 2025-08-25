from __future__ import annotations

import grpc
import asyncio
from typing import AsyncIterator, Any

from . import bentoml_service_v1alpha1_pb2 as pb
from . import bentoml_service_v1alpha1_pb2_grpc as services

class BentoMlGrpcClient:
    """
    A gRPC client for BentoML v1alpha1 BentoService.
    """

    def __init__(self, host: str, port: int | None = None, channel: grpc.aio.Channel | None = None):
        """
        Initialize the BentoML gRPC client.

        Args:
            host: The host address of the gRPC server.
            port: The port of the gRPC server. Required if 'channel' is not provided.
            channel: An existing grpc.aio.Channel. If provided, 'host' and 'port' are ignored.
        """
        if channel:
            self._channel = channel
        else:
            if port is None:
                raise ValueError("Either 'channel' or 'port' must be provided.")
            self._address = f"{host}:{port}"
            # TODO(PROTOCOL_BUFFERS_OVER_CHANNEL_SIZE_LIMIT): configure max message length
            # See https://github.com/grpc/grpc/blob/master/include/grpc/impl/codegen/grpc_types.h#L320
            # options.append(("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH))
            # options.append(("grpc.max_send_message_length", MAX_MESSAGE_LENGTH))
            self._channel = grpc.aio.insecure_channel(self._address)
        
        self._stub = services.BentoServiceStub(self._channel)

    async def health_check(self) -> Any:
        """
        Performs a health check on the gRPC server.
        This typically requires grpc_health_checking to be installed and configured on the server.
        For now, this is a placeholder or would need to call a specific health check RPC if available.
        The standard health check service might not be directly exposed via BentoServiceStub.
        """
        # This is a simplified check; real health check would use HealthStub
        # from grpc_health.v1.health_pb2_grpc import HealthStub
        # health_stub = HealthStub(self._channel)
        # request = health_pb2.HealthCheckRequest(service="bentoml.grpc.v1alpha1.BentoService")
        # return await health_stub.Check(request)
        print("Health check: Channel connectivity check.")
        try:
            # Try to connect and see if the channel is ready
            # This is a very basic check, not a formal gRPC health check
            await self._channel.channel_ready()
            return "Channel is ready."
        except grpc.aio.AioRpcError as e:
            return f"Channel is not ready: {e.code()}"


    async def call_stream(self, data: str) -> AsyncIterator[pb.Response]:
        """
        Calls the CallStream RPC method on the server.

        Args:
            data: The string data to send in the request.

        Returns:
            An async iterator yielding Response messages from the server.
        """
        request = pb.Request(data=data)
        try:
            async for response in self._stub.CallStream(request):
                yield response
        except grpc.aio.AioRpcError as e:
            # Handle potential errors, e.g., server unavailable, etc.
            print(f"gRPC call failed: {e.code()} - {e.details()}")
            # Depending on desired error handling, you might raise an exception here
            # or yield some error indication. For now, just printing and stopping iteration.
            return

    async def close(self):
        """
        Closes the gRPC channel.
        """
        if self._channel:
            await self._channel.close()

async def main():
    """
    Example usage of the BentoMlGrpcClient.
    Assumes a gRPC server is running on localhost:50051.
    """
    # Example: Start the server from src/bentoml/grpc/v1alpha1/server.py in a separate terminal
    # python src/bentoml/grpc/v1alpha1/server.py

    client = BentoMlGrpcClient(host="localhost", port=50051)
    
    print("--- Health Check ---")
    health_status = await client.health_check()
    print(f"Health status: {health_status}")
    print("\n--- Calling CallStream ---")
    input_data = "Hello from Python client"
    print(f"Sending data: '{input_data}'")
    
    try:
        i = 0
        async for response in client.call_stream(data=input_data):
            print(f"Response {i}: {response.data}")
            i += 1
    finally:
        await client.close()
        print("\nClient closed.")

if __name__ == "__main__":
    asyncio.run(main())
