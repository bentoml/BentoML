import asyncio
import grpc
from . import bentoml_service_v1alpha1_pb2 as pb
from . import bentoml_service_v1alpha1_pb2_grpc as services

class BentoServiceImpl(services.BentoServiceServicer):
    async def CallStream(self, request: pb.Request, context: grpc.aio.ServicerContext) -> pb.Response:
        """
        A streaming RPC method that accepts a Request message and returns a stream of Response messages.
        """
        print(f"Received request: {request.data}")
        for i in range(3):
            await asyncio.sleep(0.5)
            yield pb.Response(data=f"Response {i} for request: {request.data}")

async def start_grpc_server(port: int, service_instances):
    server = grpc.aio.server()
    services.add_BentoServiceServicer_to_server(BentoServiceImpl(), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f"gRPC server started on port {port}")
    await server.wait_for_termination()

if __name__ == "__main__":
    # This is for testing purposes only
    async def main():
        await start_grpc_server(50051, [])
    asyncio.run(main())
