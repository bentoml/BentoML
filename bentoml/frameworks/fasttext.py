import os

from bentoml.exceptions import MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv


class FasttextModelArtifact(BentoServiceArtifact):
    """
    Artifact class for saving and loading fasttext models

    Args:
        name (str): Name for the artifact

    Raises:
        MissingDependencyError: fasttext package is required for FasttextModelArtifact

    Example usage:

    >>> import fasttext
    >>> # prepare training data and store to file
    >>> training_data_file = 'training-data-file.train'
    >>> model = fasttext.train_supervised(input=training_data_file)
    >>>
    >>> import bentoml
    >>> from bentoml.adapters JsonInput
    >>> from bentoml.frameworks.fasttext import FasttextModelArtifact
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts([FasttextModelArtifact('model')])
    >>> class FasttextModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=JsonInput(), batch=False)
    >>>     def predict(self, parsed_json):
    >>>         # K is the number of labels that successfully were predicted,
    >>>         # among all the real labels
    >>>         return self.artifacts.model.predict(parsed_json['text'], k=5)
    >>>
    >>> svc = FasttextModelService()
    >>> svc.pack('model', model)
    """

    def __init__(self, name: str):
        super().__init__(name)

        self._model = None

    def set_dependencies(self, env: BentoServiceEnv):
        if env._infer_pip_packages:
            env.add_pip_packages(["fasttext"])

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def pack(self, fasttext_model, metadata=None):  # pylint:disable=arguments-renamed
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
