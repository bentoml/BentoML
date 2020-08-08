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
import shutil
import logging

from bentoml.artifact import BentoServiceArtifact
from bentoml.exceptions import MissingDependencyException, InvalidArgument
from bentoml.service_env import BentoServiceEnv

logger = logging.getLogger(__name__)


def _import_fastai2_module():
    try:
        import fastai2.basics
    except ImportError:
        raise MissingDependencyException(
            "fastai2 package is required to use "
            "bentoml.artifacts.Fastai2ModelArtifact"
        )

    return fastai2


class Fastai2ModelArtifact(BentoServiceArtifact):
    """Saving and Loading FastAI2 Model

    Args:
        name (str): Name for the fastai2 model

    Raises:
        MissingDependencyException: Require fastai2 package for Fastai2 model artifact
        InvalidArgument: invalid argument type, model being packed must be instance of
            fastai2.basics.Learner
    """

    def __init__(self, name):
        super(Fastai2ModelArtifact, self).__init__(name)
        self._file_name = name + '.pkl'
        self._model = None

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self._file_name)

    def pack(self, model):  # pylint:disable=arguments-differ
        fastai2_module = _import_fastai2_module()

        if not isinstance(model, fastai2_module.basics.Learner):
            raise InvalidArgument(
                "Expect `model` argument to be `fastai2.basics.Learner` instance"
            )

        self._model = model
        return self

    def load(self, path):
        fastai2_module = _import_fastai2_module()

        model = fastai2_module.basics.load_learner(path + '/' + self._file_name)
        return self.pack(model)

    def set_dependencies(self, env: BentoServiceEnv):
        logger.warning(
            "BentoML by default does not include spacy and torchvision package when "
            "using Fastai2ModelArtifact. To make sure BentoML bundle those packages if "
            "they are required for your model, either import those packages in "
            "BentoService definition file or manually add them via "
            "`@env(pip_dependencies=['torchvision'])` when defining a BentoService"
        )
        env.add_pip_dependencies_if_missing(['torch', "fastcore", "fastai2"])

    def save(self, dst):
        self._model.export(fname=self._file_name)

        shutil.copyfile(
            os.path.join(self._model.path, self._file_name), self._model_file_path(dst),
        )

    def get(self):
        return self._model
