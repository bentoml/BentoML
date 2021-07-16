import json
from typing import Sequence

from ..adapters.base_output import regroup_return_value
from ..adapters.json_output import JsonOutput
from ..adapters.utils import TfTensorJsonEncoder
from ..types import InferenceError, InferenceResult, InferenceTask
from ..utils.lazy_loader import LazyLoader

np = LazyLoader("np", globals(), "numpy")


def tf_to_numpy(tensor):
    """
    Tensor -> ndarray
    List[Tensor] -> tuple[ndarray]
    """
    import tensorflow as tf

    if isinstance(tensor, (list, tuple)):
        return tuple(tf_to_numpy(t) for t in tensor)

    if tf.__version__.startswith("1."):
        with tf.compat.v1.Session():
            return tensor.numpy()
    else:
        return tensor.numpy()


class TfTensorOutput(JsonOutput):
    """
    Output adapters converts returns of user defined API function into specific output,
    such as HTTP response, command line stdout or AWS Lambda event object.

    Args:
        cors (str): DEPRECATED. Moved to the configuration file.
            The value of the Access-Control-Allow-Origin header set in the
            HTTP/AWS Lambda response object. If set to None, the header will not be set.
            Default is None.
        ensure_ascii(bool): Escape all non-ASCII characters. Default False.
    """

    BATCH_MODE_SUPPORTED = True

    @property
    def pip_dependencies(self):
        """
        :return: List of PyPI package names required by this OutputAdapter
        """
        return ["tensorflow"]

    def pack_user_func_return_value(
        self, return_result, tasks: Sequence[InferenceTask],
    ) -> Sequence[InferenceResult[str]]:
        rv = []
        results = tf_to_numpy(return_result)
        for result, _ in regroup_return_value(results, tasks):
            try:
                result_str = json.dumps(result, cls=TfTensorJsonEncoder)
                rv.append(InferenceResult(data=result_str, http_status=200))
            except Exception as e:  # pylint: disable=broad-except
                rv.append(InferenceError(err_msg=str(e), http_status=500))
        return rv
