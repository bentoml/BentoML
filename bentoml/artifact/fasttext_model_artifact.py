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
            import fasttext
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
            import fasttext
        except ImportError:
            raise MissingDependencyException(
                "fasttext package is required to use FasttextModelArtifact"
            )
        self._model = model

    def get(self):
        return self._model

    def save(self, dst):
        self._model.save_model(self.spec._model_file_path(dst))
