import bentoml

square_runner = bentoml.picklable_model.get(
    "my_python_model:latest").to_runner()

svc = bentoml.Service("simple_square_svc", runners=[square_runner])


@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def square(input_arr):
    return square_runner.run(input_arr)
