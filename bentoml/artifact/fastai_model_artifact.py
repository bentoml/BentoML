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


def _import_fastai_module():
    try:
        import fastai.basic_train
    except ImportError:
        raise MissingDependencyException(
            "fastai package is required to use " "bentoml.artifacts.FastaiModelArtifact"
        )

    return fastai


class FastaiModelArtifact(BentoServiceArtifact):
    """Saving and Loading FastAI Model

    Args:
        name (str): Name for the fastai model

    Raises:
        MissingDependencyException: Require fastai package to use Fast ai model artifact
        InvalidArgument: invalid argument type, model being packed must be instance of
            fastai.basic_train.Learner

    Example usage:

    >>> from fastai.tabular import *
    >>>
    >>> # prepare data
    >>> data = TabularList.from_df(...)
    >>> learn = tabular_learner(data, ...)
    >>> # train model
    >>>
    >>> import bentoml
    >>> from bentoml.adapters import DataframeInput
    >>> from bentoml.artifact import FastaiModelArtifact
    >>>
    >>> @bentoml.artifacts([FastaiModelArtifact('model')])
    >>> @bentoml.env(auto_pip_dependencies=True)
    >>> class FastaiModelService(bentoml.BentoService):
    >>>
    >>>     @api(input=DataframeInput())
    >>>     def predict(self, df):
    >>>         results = []
    >>>         for _, row in df.iterrows():
    >>>             prediction = self.artifacts.model.predict(row)
    >>>             results.append(prediction[0].obj)
    >>>         return results
    >>>
    >>> svc = FastaiModelService()
    >>>
    >>> # Pack fastai basic_learner directly
    >>> svc.pack('model', learn)
    """

    def __init__(self, name):
        super(FastaiModelArtifact, self).__init__(name)
        self._file_name = name + '.pkl'
        self._model = None

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self._file_name)

    def pack(self, model):  # pylint:disable=arguments-differ
        fastai_module = _import_fastai_module()

        if not isinstance(model, fastai_module.basic_train.Learner):
            raise InvalidArgument(
                "Expect `model` argument to be `fastai.basic_train.Learner` instance"
            )

        self._model = model
        return self

    def load(self, path):
        fastai_module = _import_fastai_module()

        model = fastai_module.basic_train.load_learner(path, self._file_name)
        return self.pack(model)

    def set_dependencies(self, env: BentoServiceEnv):
        logger.warning(
            "BentoML by default does not include spacy and torchvision package when "
            "using FastaiModelArtifact. To make sure BentoML bundle those packages if "
            "they are required for your model, either import those packages in "
            "BentoService definition file or manually add them via "
            "`@env(pip_dependencies=['torchvision'])` when defining a BentoService"
        )
        env.add_pip_dependencies_if_missing(['torch', "fastai"])

    def save(self, dst):
        self._model.export(file=self._file_name)

        shutil.copyfile(
            os.path.join(self._model.path, self._file_name), self._model_file_path(dst),
        )

    def get(self):
        return self._model
