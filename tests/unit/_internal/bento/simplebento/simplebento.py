import bentoml

# import bentoml.sklearn
# from bentoml.io import NumpyNdarray

# iris_model_runner = bentoml.sklearn.get('iris_classifier:latest').to_runner()
svc = bentoml.Service(
    "test.simplebento",
    # runners=[iris_model_runner]
)

# @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
# def predict(request_data: np.ndarray):
#     return iris_model_runner.predict(request_data)

# For simple use cases, only models list is required:
# svc.bento_options.models = []
# svc.bento_files.include = ["*"]
# svc.bento_env.pip_install = "./requirements.txt"
