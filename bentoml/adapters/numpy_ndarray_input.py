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

from typing import Sequence, Tuple

from bentoml.adapters.string_input import StringInput
from bentoml.exceptions import MissingDependencyException
from bentoml.types import InferenceTask
from bentoml.utils.lazy_loader import LazyLoader

numpy = LazyLoader('numpy', globals(), 'numpy')

DataFrameTask = InferenceTask[str]
ApiFuncArgs = Tuple['numpy.ndarray']


class NumpyNdarrayInput(StringInput):
    """
    Convert various inputs(HTTP, Aws Lambda or CLI) to numpy ndarray, passing it to
    API functions.

    Parameters
    ----------
    dtype : dict, default None
        If is None, infer dtypes; if a dict of column to dtype, then use those.

    Raises
    -------
    ValueError: Incoming data is missing required columns in dtype

    ValueError: Incoming data format can not be handled. Only json and csv

    Examples
    -------
    Example Service:

    .. code-block:: python

        from bentoml import env, artifacts, api, BentoService
        from bentoml.adapters import NumpyNdarrayInput
        from bentoml.frameworks.sklearn import SklearnModelArtifact

        @env(infer_pip_packages=True)
        @artifacts([SklearnModelArtifact('model')])
        class IrisClassifier(BentoService):

            @api(
                input=NumpyNdarrayInput(dtype='int32'),
                batch=True,
            )
            def predict(self, ndarray):
                # Optional pre-processing, post-processing code goes here
                return self.artifacts.model.predict(ndarray)

    Query with HTTP request::

        curl -i \\
          --header "Content-Type: application/json" \\
          --request POST \\
          --data '[[1,2,3,4,5]]' \\
          localhost:5000/predict

    Query with CLI command::

        bentoml run IrisClassifier:latest predict --input \\
          '[[1,2,3,4,5]]'


    """

    SINGLE_MODE_SUPPORTED = False

    def __init__(
        self, dtype: str = None, **base_kwargs,
    ):
        super().__init__(**base_kwargs)

        # Verify numpy imported properly and retry import if it has failed initially
        if numpy is None:
            raise MissingDependencyException(
                "Missing required dependency 'numpy' for NumpyNdarrayInput, install "
                "with `pip install numpy`"
            )
        if isinstance(dtype, (list, tuple)):
            self.dtype = dict((index, dtype) for index, dtype in enumerate(dtype))
        else:
            self.dtype = dtype

    @property
    def pip_dependencies(self):
        return ['numpy']

    @property
    def config(self):
        base_config = super().config
        return dict(base_config, dtype=self.dtype,)

    @property
    def request_schema(self):
        json_schema = {"schema": {"type": "object"}}
        return {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {"file": {"type": "string", "format": "binary"}},
                }
            },
            "application/json": json_schema,
            "text/csv": {"schema": {"type": "string", "format": "binary"}},
        }

    class _Undefined:
        pass

    def extract_user_func_args(
        self, tasks: Sequence[InferenceTask[str]]
    ) -> ApiFuncArgs:

        arrays = [self._Undefined] * len(tasks)

        for i, task in enumerate(tasks):
            try:
                arrays[i] = numpy.array(task.data, dtype=self.dtype)
            except ValueError:
                task.discard(
                    err_msg=f"invalid literal for {self.dtype} with value: {task.data}",
                    http_status=400,
                )

        return (numpy.stack(tuple(a for a in arrays if a is not self._Undefined)),)
