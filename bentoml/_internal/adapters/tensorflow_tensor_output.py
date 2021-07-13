# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Sequence

from bentoml.adapters.base_output import regroup_return_value
from bentoml.adapters.json_output import JsonOutput
from bentoml.adapters.utils import TfTensorJsonEncoder
from bentoml.types import InferenceError, InferenceResult, InferenceTask
from bentoml.utils.lazy_loader import LazyLoader

np = LazyLoader('np', globals(), 'numpy')


def tf_to_numpy(tensor):
    '''
    Tensor -> ndarray
    List[Tensor] -> tuple[ndarray]
    '''
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
        return ['tensorflow']

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
