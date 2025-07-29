import asyncio
import time
import uuid

import grpc

# Import generated gRPC stubs
from generated import example_service_pb2
from generated import example_service_pb2_grpc


async def run_client():
    # Target server address
    target_address = (
        "localhost:50051"  # Default gRPC port for BentoML, adjust if necessary
    )

    # Create a channel
    async with grpc.aio.insecure_channel(target_address) as channel:
        # Create a stub (client)
        stub = example_service_pb2_grpc.SimpleStreamingServiceStub(channel)

        # Prepare a request message
        client_message_text = "Hello, stream!"
        request_message = example_service_pb2.ChatMessage(
            message_id=str(uuid.uuid4()),  # Unique ID for the message
            text=client_message_text,
            timestamp=int(time.time() * 1000),
        )

        print(f"Client sending: {client_message_text}")

        try:
            # Call the Chat RPC method and iterate through the streamed responses
            async for response in stub.Chat(request_message):
                print(
                    f"Server says: {response.text} (ID: {response.message_id}, TS: {response.timestamp})"
                )

            print("Stream finished.")

        except grpc.aio.AioRpcError as e:
            print(f"gRPC call failed: {e.code()} - {e.details()}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    print("Starting gRPC client example...")
    asyncio.run(run_client())
