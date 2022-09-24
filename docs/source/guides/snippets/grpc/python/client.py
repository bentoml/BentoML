import asyncio

import grpc

from bentoml.grpc.utils import import_generated_stubs

pb, services = import_generated_stubs()


async def run():
    async with grpc.aio.insecure_channel("localhost:3000") as channel:
        stub = services.BentoServiceStub(channel)
        req = await stub.Call(
            request=pb.Request(
                api_name="classify",
                ndarray=pb.NDArray(
                    dtype=pb.NDArray.DTYPE_FLOAT,
                    shape=(1, 4),
                    float_values=[5.9, 3, 5.1, 1.8],
                ),
            )
        )
    print(req)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()
        assert loop.is_closed()
