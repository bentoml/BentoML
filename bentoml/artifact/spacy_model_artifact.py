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

import logging
import os

from bentoml.artifact import BentoServiceArtifact
from bentoml.exceptions import MissingDependencyException, InvalidArgument
from bentoml.service_env import BentoServiceEnv

logger = logging.getLogger(__name__)


class SpacyModelArtifact(BentoServiceArtifact):
    """
    Abstraction for saving/loading spacy models
    with to_disk and spacy.util.load_model methods.
    Args:
        name (string): name of the artifact
    Raises:
        MissingDependencyException: spacy package is required for SpacyModelArtifact
        InvalidArgument: invalid argument type, model being packed must be instance of
            spacy.language.Language
    Example usage:

    >>> import spacy
    >>> # TRAIN MODEL WITH DATA
    >>> TRAIN_DATA = [
    >>>     ("Uber blew through $1 million a week", {"entities": [(0, 4, "ORG")]}),
    >>>     ("Google rebrands its business apps", {"entities": [(0, 6, "ORG")]})]
    >>>
    >>> nlp = spacy.blank("en")
    >>> optimizer = nlp.begin_training()
    >>> for i in range(20):
    >>>     random.shuffle(TRAIN_DATA)
    >>>     for text, annotations in TRAIN_DATA:
    >>>         nlp.update([text], [annotations], sgd=optimizer)
    >>> nlp.to_disk("/model")
    >>>
    >>> import bentoml
    >>> from bentoml.adapters import JsonInput
    >>> from bentoml.artifact import SpacyModelArtifact
    >>>
    >>> @bentoml.env(auto_pip_dependencies=True)
    >>> @bentoml.artifacts([SpacyModelArtifact('nlp')])
    >>> class SpacyModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=JsonInput())
    >>>     def predict(self, parsed_json):
    >>>         outputs = self.artifacts.nlp(parsed_json['text'])
    >>>         return outputs
    >>>
    >>>
    >>> svc = SpacyModelService()
    >>>
    >>> # Spacy model can be packed directly.
    >>> svc.pack('nlp', nlp)
    """

    def __init__(self, name):
        super(SpacyModelArtifact, self).__init__(name)

        self._model = None

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def pack(self, model):  # pylint:disable=arguments-differ
        try:
            import spacy
        except ImportError:
            raise MissingDependencyException(
                "spacy package is required to use SpacyModelArtifact"
            )

        if not isinstance(model, spacy.language.Language):
            raise InvalidArgument(
                "SpacyModelArtifact can only pack type 'spacy.language.Language'"
            )

        self._model = model
        return self

    def load(self, path):
        try:
            import spacy
        except ImportError:
            raise MissingDependencyException(
                "spacy package is required to use SpacyModelArtifact"
            )

        model = spacy.util.load_model(self._file_path(path))

        if not isinstance(model, spacy.language.Language):
            raise InvalidArgument(
                "Expecting SpacyModelArtifact loaded object type to be "
                "'spacy.language.Language' but actually it is {}".format(type(model))
            )

        return self.pack(model)

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_dependencies_if_missing(['spacy'])

    def get(self):
        return self._model

    def save(self, dst):
        path = self._file_path(dst)
        return self._model.to_disk(path)
