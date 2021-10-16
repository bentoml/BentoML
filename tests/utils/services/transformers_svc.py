import bentoml.transformers
from bentoml import Service
from bentoml.io import NumpyNdarray

gpt_runner = bentoml.transformers.load_runner(tag="gpt2_tests", tasks="text-generation")

svc = Service("gpt2-tg", runners=[gpt_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray(), route="/predict")
def predict():
    ...


@svc.api(input=NumpyNdarray(), output=NumpyNdarray(), route="/")
def preprocess():
    ...
