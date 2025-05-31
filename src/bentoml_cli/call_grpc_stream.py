from __future__ import annotations

import asyncio
import click
import sys

from bentoml.grpc.v1alpha1.client import BentoMlGrpcClient

@click.command(name="call-grpc-stream")
@click.option(
    "--host", 
    type=click.STRING, 
    default="localhost", 
    help="The host address of the gRPC server.",
    show_default=True,
)
@click.option(
    "--port", 
    type=click.INT, 
    default=50051, 
    help="The port of the gRPC server.",
    show_default=True,
)
@click.option(
    "--data", 
    type=click.STRING, 
    required=True, 
    help="The string data to send to the CallStream method."
)
def call_grpc_stream_command(host: str, port: int, data: str) -> None:
    """
    Call the CallStream gRPC method of a BentoML v1alpha1 service.
    This command connects to a gRPC server and invokes the CallStream method,
    printing each streamed response to standard output.
    """

    async def _main():
        client = BentoMlGrpcClient(host=host, port=port)
        try:
            print(f"Connecting to gRPC server at {host}:{port}...")
            print(f"Sending data: '{data}' to CallStream...")
            print("--- Streamed Responses ---")
            async for response in client.call_stream(data=data):
                # Assuming response.data is the field to print.
                # Adjust if your Response message structure is different.
                print(response.data)
        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)
        finally:
            if client:
                await client.close()
            print("------------------------")
            print("Connection closed.")

    asyncio.run(_main())

if __name__ == "__main__":
    # This allows running the command directly for testing if needed,
    # though it's primarily designed to be invoked via the bentoml CLI group.
    call_grpc_stream_command()
