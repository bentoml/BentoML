import bentoml
from bentoml.io import NumpyNdarray

reg_runner = bentoml.sklearn.get("linear_reg:latest").to_runner()

svc = bentoml.Service("linear_regression", runners=[reg_runner])

input_spec = NumpyNdarray(dtype="int", shape=(-1, 2))


@svc.api(input=input_spec, output=NumpyNdarray())
async def predict(input_arr):
    return await reg_runner.predict.async_run(input_arr)
