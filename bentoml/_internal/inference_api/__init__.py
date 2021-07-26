def api(input, output):  # pylint: disable=unused-argument
    """
    decorator for declaring a instance method in a bentoml.Service class as an
    InferenceAPI
    """


def batch_api(
    input, output, max_latency, max_batch_size
):  # pylint: disable=unused-argument
    """
    decorator for declaring a instance method in a bentoml.Service class as an
    BatchInferenceAPI
    """
