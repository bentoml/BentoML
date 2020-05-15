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

from bentoml.artifact import BentoServiceArtifact, BentoServiceArtifactWrapper
from bentoml.exceptions import MissingDependencyException, InvalidArgument

logger = logging.getLogger(__name__)


class OnnxModelArtifact(BentoServiceArtifact):
    """Abstraction for saving/loading onnx model

    Args:
        name (string): name of the artifact
    Raises:
        MissingDependencyException: onnx package is required for OnnxModelArtifact
    """

    def __init__(self, name, backend='onnx'):
        super(OnnxModelArtifact, self).__init__(name)
        self.model_extension = '.onnx'
        self.backend = backend
        # throw if training_framework is empty

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + '.onnx')

    def pack(self, model, training_framework=None, initial_types=None):  # pylint:disable=arguments-differ
        return _OnnxModelArtifactWrapper(
            self, model, training_framework, initial_types
        )

    def load(self, path):
        if self.backend == 'onnx':
            inference_model = OnnxRuntimeSession(self._model_file_path(path))
        else:
            raise NotImplementedError('not now')
        return self.pack(inference_model)

    @property
    def pip_dependencies(self):
        dependencies = ['onnx']
        if self.backend == 'onnx':
            dependencies.append('onnxruntime')
        return dependencies


class _OnnxModelArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, model, training_framework, initial_types=None):
        super(_OnnxModelArtifactWrapper, self).__init__(spec)
        try:
            import onnx
        except ImportError:
            raise MissingDependencyException(
                "onnx package is required to use OnnxModelArtifact"
            )
        # check instance
        self._model = model
        self.training_framework = training_framework
        self.initial_types = initial_types

    def get(self):
        return self._model

    def save(self, dst):
        try:
            import onnxmltools
        except ImportError:
            raise MissingDependencyException(
                "onnx package is required to use OnnxModelArtifact"
            )

        logger.info(self.training_framework)
        logger.info(self.initial_types)
        if self.training_framework == 'scikit-learn':
            try:
                import skl2onnx
            except ImportError:
                raise MissingDependencyException(
                    'skl2onnx package is required for converting sk model to ONNX model'
                )
            converted_model = onnxmltools.convert_sklearn(
                model=self._model, initial_types=self.initial_types
            )
        elif self.training_framework == 'pytorch':
            raise NotImplementedError
        elif self.training_framework == 'tensorflow':
            try:
                import tf2onnx
            except ImportError:
                raise MissingDependencyException(
                    'tf2onnx is required for converting Tensorflow model to ONNX model'
                )
            converted_model = onnxmltools.convert_tensorflow(
                frozen_graph_def=self._model
            )
        elif self.training_framework == 'keras':
            converted_model = onnxmltools.convert_keras(
                model=self._model, initial_types=self.initial_types
            )
        elif self.training_framework == 'lightgbm':
            converted_model = onnxmltools.convert_lightgbm(
                model=self._model, initial_types=self.initial_types
            )
        elif self.training_framework == 'xgboost':
            converted_model = onnxmltools.convert_xgboost(
                model = self._model, initial_types=self.initial_types
            )
        # not release on pypi yet
        # elif self.training_framework == 'h2o':
        #     converted_model = onnxmltools.convert_h2o(
        #         model=self._model, initial_types=self.initial_types
        #     )
        elif self.training_framework == 'coreml':
            try:
                import coremltools
            except ImportError:
                raise MissingDependencyException(
                    'coremltools is required for converting CoreML model to ONNX model'
                )
            converted_model = onnxmltools.convert_coreml(
                model=self._model, initial_types=self.initial_types
            )
        else:
            raise InvalidArgument('training framework not supported')

        return onnxmltools.utils.save_model(converted_model, self.spec._model_file_path(dst))


class OnnxRuntimeSession:
    def __init__(self, model_path):
        try:
            import onnxruntime
        except ImportError:
            raise MissingDependencyException(
                'onnxruntime package is required for inferencing using ONNX runtime'
            )

        self.model_path = model_path
        self.session = onnxruntime.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_data):
        return self.session.run([self.output_name], {self.input_name: input_data})
