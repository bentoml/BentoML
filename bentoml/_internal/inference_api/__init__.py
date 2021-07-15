def api(input, output):
    """
    decorator for declaring a instance method in a bentoml.Service class as an
    InferenceAPI
    """
    pass


def batch_api(input, output, max_latency, max_batch_size):
    """
    decorator for declaring a instance method in a bentoml.Service class as an
    BatchInferenceAPI
    """
