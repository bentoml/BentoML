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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pathlib
import shutil

from bentoml.artifact import BentoServiceArtifact, BentoServiceArtifactWrapper
from bentoml.exceptions import MissingDependencyException, InvalidArgument

logger = logging.getLogger(__name__)


def _is_file_like(p):
    return (
        isinstance(p, (str, bytes, pathlib.PurePath, os.PathLike))
        and os.path.isfile(p)
    )


def _load_onnx_saved_model(path):
    try:
        import onnxruntime
    except ImportError:
        raise MissingDependencyException(
            'onnxruntime package is required for inferencing using ONNX runtime'
        )
    model = onnxruntime.InferenceSession(path)
    return model


class OnnxModelArtifact(BentoServiceArtifact):
    """Abstraction for saving/loading onnx model

    Args:
        name (string): Name of the artifact
        backend (string): Name of ONNX inference runtime. [onnx]
    Raises:
        MissingDependencyException: onnx package is required for packing a ModelProto
                                    object
        NotImplementedError: {backend} as onnx runtime is not supported at the moment

    Example usage:

    >>>
    >>>
    >>>
    >>> # Train a model.
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> clr = RandomForestClassifier()
    >>> clr.fit(X_train, y_train)

    >>> # Convert into ONNX format
    >>> from skl2onnx import convert_sklearn
    >>> from skl2onnx.common.data_types import FloatTensorType
    >>> initial_type = [('float_input', FloatTensorType([None, 4]))]
    >>>
    >>> onnx_model = convert_sklearn(clr, initial_types=initial_type)
    >>> with open("rf_iris.onnx", "wb") as f:
    >>>     f.write(onnx_model.SerializeToString())
    >>>
    >>>
    >>> import bentoml
    >>> from bentoml.artifact import OnnxModelArtifact
    >>> from bentoml.handlers import DataframeHandler
    >>>
    >>> @bentoml.env(auto_pip_dependencies=True)
    >>> @bentoml.artifact([OnnxModelArtifact('model')])
    >>> class OnnxIrisClassifierService(bentoml.BentoService):
    >>>     @bentoml.api(DataframeHandler)
    >>>     def predict(self, df):
    >>>         input_name = self.artifacts.model.get_inputs()[0].name
    >>>         output_name = self.artifacts.model.get_outputs()[0].name
    >>>         return self.artifacts.model.run([output_name], {input_name: df})[0]
    >>>
    >>> svc = OnnxIrisClassifierService()
    >>>
    >>> # Option one: pack with path to model on local system
    >>> svc.pack('model', './rf_iris.onnx')
    >>>
    >>> # Option two: pack with ONNX model object
    >>> # svc.pack('model', onnx_model)
    >>>
    >>> # Save BentoService
    >>> svc.save()
    """

    def __init__(self, name, backend='onnx'):
        super(OnnxModelArtifact, self).__init__(name)
        self.backend = backend

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + '.onnx')

    def pack(self, obj):  # pylint:disable=arguments-differ
        if _is_file_like(obj):
            return _ExportedOnnxModelArtifact(self, obj)
        else:
            try:
                import onnx
            except ImportError:
                raise MissingDependencyException(
                    'onnx package is required for packing a ModelProto object'
                )
            if not isinstance(self._model, onnx.ModelProto):
                raise InvalidArgument('Object is not an instance of onnx.ModelProto')
            return _OnnxModelArtifactWrapper(self, obj)

    def load(self, path):
        if self.backend == 'onnx':
            inference_model = _load_onnx_saved_model(self._model_file_path(path))
        else:
            raise NotImplementedError(
                f'{self.backend} as onnx runtime is not supported at the moment'
            )
        return self.pack(inference_model)

    @property
    def pip_dependencies(self):
        dependencies = []
        if self.backend == 'onnx':
            dependencies.append('onnxruntime')
        return dependencies


class _OnnxModelArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, model):
        super(_OnnxModelArtifactWrapper, self).__init__(spec)
        self._model = model

    def get(self):
        return self._model

    def save(self, dst):
        try:
            import onnx
        except ImportError:
            raise MissingDependencyException(
                'onnx package is required to save onnx.ModelProto object'
            )
        if not isinstance(self._model, onnx.ModelProto):
            raise InvalidArgument('Object is not an instance of onnx.ModelProto')
        return onnx.save(self._model, self.spec._model_file_path(dst))


class _ExportedOnnxModelArtifact(BentoServiceArtifactWrapper):
    def __init__(self, spec, path):
        super(_ExportedOnnxModelArtifact, self).__init__(spec)

        self.path = path
        self.model = None

    def save(self, dst):
        shutil.copyfile(self.path, self.spec._model_file_path(dst))

    def get(self):
        if not self.model:
            self.model = _load_onnx_saved_model(self.spec._model_file_path(self.path))
        return self.model
