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

import os

from bentoml.artifact import BentoServiceArtifact
from bentoml.exceptions import MissingDependencyException
from bentoml.service_env import BentoServiceEnv


class FasttextModelArtifact(BentoServiceArtifact):
    """
    Abstraction for saving/loading fasttext models

    Args:
        name (str): Name for the artifact

    Raises:
        MissingDependencyError: fasttext package is required for FasttextModelArtifact

    Example usage:

    >>> import fasttext
    >>> # prepare training data and store to file
    >>> training_data_file = 'trainging-data-file.train'
    >>> model = fasttext.train_supervised(input=training_data_file)
    >>>
    >>> import bentoml
    >>> from bentoml.adapters JsonInput
    >>> from bentoml.artifact import FasttextModelArtifact
    >>>
    >>> @bentoml.env(auto_pip_dependencies=True)
    >>> @bentoml.artifacts([FasttextModelArtifact('model')])
    >>> class FasttextModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=JsonInput())
    >>>     def predict(self, parsed_json):
    >>>         # K is the number of labels that successfully were predicted,
    >>>         # among all the real labels
    >>>         return self.artifacts.model.predict(parsed_json['text'], k=5)
    >>>
    >>> svc = FasttextModelService()
    >>> svc.pack('model', model)
    """

    def __init__(self, name):
        super(FasttextModelArtifact, self).__init__(name)

        self._model = None

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_dependencies_if_missing(["fasttext"])

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def pack(self, fasttext_model):  # pylint:disable=arguments-differ
        try:
            import fasttext  # noqa # pylint: disable=unused-import
        except ImportError:
            raise MissingDependencyException(
                "fasttext package is required to use FasttextModelArtifact"
            )
        self._model = fasttext_model
        return self

    def load(self, path):
        try:
            import fasttext  # noqa # pylint: disable=unused-import
        except ImportError:
            raise MissingDependencyException(
                "fasttext package is required to use FasttextModelArtifact"
            )
        model = fasttext.load_model(self._model_file_path(path))
        return self.pack(model)

    def get(self):
        return self._model

    def save(self, dst):
        self._model.save_model(self._model_file_path(dst))
