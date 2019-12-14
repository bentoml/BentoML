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

import os
import shutil

from bentoml.artifact import BentoServiceArtifact, BentoServiceArtifactWrapper
from bentoml.exceptions import MissingDependencyException, InvalidArgument


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
    """

    def __init__(self, name):
        super(FastaiModelArtifact, self).__init__(name)
        self._file_name = name + '.pkl'

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self._file_name)

    def pack(self, model):  # pylint:disable=arguments-differ
        fastai_module = _import_fastai_module()

        if not isinstance(model, fastai_module.basic_train.Learner):
            raise InvalidArgument(
                "Expect `model` argument to be `fastai.basic_train.Learner` instance"
            )

        return _FastaiModelArtifactWrapper(self, model)

    def load(self, path):
        fastai_module = _import_fastai_module()

        model = fastai_module.basic_train.load_learner(path, self._file_name)
        return self.pack(model)


class _FastaiModelArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, model):
        super(_FastaiModelArtifactWrapper, self).__init__(spec)

        self._model = model

    def save(self, dst):
        self._model.export(file=self.spec._file_name)

        shutil.copyfile(
            os.path.join(self._model.path, self.spec._file_name),
            self.spec._model_file_path(dst),
        )

    def get(self):
        return self._model
