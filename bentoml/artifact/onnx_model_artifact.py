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


def _is_path_like(p):
    return isinstance(p, (str, bytes, pathlib.PurePath, os.PathLike))


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
        name (string): name of the artifact
    Raises:
        MissingDependencyException: onnx package is required for OnnxModelArtifact
    """

    def __init__(self, name, backend='onnx'):
        super(OnnxModelArtifact, self).__init__(name)
        self.backend = backend

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + '.onnx')

    def pack(self, obj):  # pylint:disable=arguments-differ
        if _is_path_like(obj):
            return _ExportedOnnxModelArtifact(self, obj)
        return _OnnxModelArtifactWrapper(self, obj)

    def load(self, path):
        if self.backend == 'onnx':
            inference_model = _load_onnx_saved_model(self._model_file_path(path))
        else:
            raise NotImplementedError('not now bro')
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
            raise MissingDependencyException('TODO')
        if not isinstance(self._model, onnx.ModelProto):
            raise InvalidArgument('TODO')
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


# class OnnxRuntimeSession:
#     def __init__(self, model_path):
#         try:
#             import onnxruntime
#         except ImportError:
#             raise MissingDependencyException(
#                 'onnxruntime package is required for inferencing using ONNX runtime'
#             )
#
#         self.model_path = model_path
#         self.session = onnxruntime.InferenceSession(self.model_path)
#         self.input_name = self.session.get_inputs()[0].name
#         self.output_name = self.session.get_outputs()[0].name
#
#     def predict(self, input_data):
#         return self.session.run([self.output_name], {self.input_name: input_data})
