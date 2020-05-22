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
    >>> from bentoml.handlers import JsonHandler
    >>> from bentoml.artifact import SpacyModelArtifact
    >>>
    >>> @bentoml.env(auto_pip_dependencies=True)
    >>> @bentoml.artifacts([SpacyModelArtifact('nlp')])
    >>> class SpacyModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(JsonHandler)
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

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def pack(self, model):  # pylint:disable=arguments-differ
        return _SpacyModelArtifactWrapper(self, model)

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

    @property
    def pip_dependencies(self):
        return ['spacy']


class _SpacyModelArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, model):
        super(_SpacyModelArtifactWrapper, self).__init__(spec)

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

    def get(self):
        return self._model

    def save(self, dst):
        path = os.path.join(dst, self.spec.name)
        return self._model.to_disk(path)
