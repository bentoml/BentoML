import logging
import os

from bentoml.exceptions import InvalidArgument, MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv

logger = logging.getLogger(__name__)


class SpacyModelArtifact(BentoServiceArtifact):
    """
    Artifact class for saving/loading spacy models with Language.to_disk and
    spacy.util.load_model methods

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
    >>> from bentoml.frameworks.spacy import SpacyModelArtifact
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts([SpacyModelArtifact('nlp')])
    >>> class SpacyModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=JsonInput(), batch=False)
    >>>     def predict(self, parsed_json):
    >>>         output = self.artifacts.nlp(parsed_json['text'])
    >>>         return output
    >>>
    >>>
    >>> svc = SpacyModelService()
    >>>
    >>> # Spacy model can be packed directly.
    >>> svc.pack('nlp', nlp)
    """

    def __init__(self, name):
        super().__init__(name)

        self._model = None

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def pack(self, model, metadata=None):  # pylint:disable=arguments-differ
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
        if env._infer_pip_packages:
            env.add_pip_packages(['spacy'])

    def get(self):
        return self._model

    def save(self, dst):
        path = self._file_path(dst)
        return self._model.to_disk(path)
