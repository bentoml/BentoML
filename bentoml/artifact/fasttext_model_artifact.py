from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from bentoml.artifact import BentoServiceArtifact, BentoServiceArtifactWrapper
from bentoml.exceptions import MissingDependencyException


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
    >>> from bentoml.handlers import JsonHandler
    >>> from bentoml.artifact import FasttextModelArtifact
    >>>
    >>> @bentoml.env(auto_pip_dependencies=True)
    >>> @bentoml.artifacts([FasttextModelArtifact('model')])
    >>> class FasttextModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(JsonHandler)
    >>>     def predict(self, parsed_json):
    >>>         # K is the number of labels that successfully were predicted,
    >>>         # among all the real labels
    >>>         return self.artifacts.model.predict(parsed_json['text'], k=5)
    >>>
    >>> svc = FasttextModelService()
    >>> svc.pack('model', model)
    """

    @property
    def pip_dependencies(self):
        return ["fasttext"]

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def pack(self, fasttext_model):  # pylint:disable=arguments-differ
        return _FasttextModelArtifactWrapper(self, fasttext_model)

    def load(self, path):
        try:
            import fasttext  # noqa # pylint: disable=unused-import
        except ImportError:
            raise MissingDependencyException(
                "fasttext package is required to use FasttextModelArtifact"
            )
        model = fasttext.load_model(self._model_file_path(path))
        return self.pack(model)


class _FasttextModelArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, model):
        super(_FasttextModelArtifactWrapper, self).__init__(spec)
        try:
            import fasttext  # noqa # pylint: disable=unused-import
        except ImportError:
            raise MissingDependencyException(
                "fasttext package is required to use FasttextModelArtifact"
            )
        self._model = model

    def get(self):
        return self._model

    def save(self, dst):
        self._model.save_model(self.spec._model_file_path(dst))
