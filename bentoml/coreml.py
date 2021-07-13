import logging
import os

from bentoml.exceptions import InvalidArgument, MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv

logger = logging.getLogger(__name__)


COREML_MODEL_FILE_EXTENTION = ".mlmodel"


class CoreMLModelArtifact(BentoServiceArtifact):
    """
    Artifact class for saving/loading coreml.models.MLModel objects
    with coremltools.models.MLModel.save and coremltools.models.MLModel(path).

    Args:
        name (string): name of the artifact

    Raises:
        MissingDependencyException: coremltools package required for CoreMLModelArtifact
        InvalidArgument: invalid argument type, model being packed must be instance of
            coremltools.models.MLModel

    Example usage:

    >>> import coremltools as ct
    >>> import torch.nn as nn
    >>>
    >>> class Net(nn.Module):
    >>>     def __init__(self):
    >>>         super(Net, self).__init__()
    >>>         ...
    >>>
    >>>     def forward(self, x):
    >>>         ...
    >>>
    >>> net = Net()
    >>> # Train model with data, then convert to CoreML.
    >>> model = ct.convert(net, ...)
    >>>
    >>> import bentoml
    >>> import PIL.Image
    >>> from bentoml.adapters import ImageInput
    >>> from bentoml.frameworks.coreml import CoreMLModelArtifact
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts([CoreMLModelArtifact('model')])
    >>> class CoreMLModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=ImageInput(), batch=True)
    >>>     def predict(self, imgs):
    >>>         outputs = [self.artifacts.model(PIL.Image.fromarray(_.astype("uint8")))
    >>>                    for img in imgs]
    >>>         return outputs
    >>>
    >>>
    >>> svc = CoreMLModelService()
    >>>
    >>> # Pytorch model can be packed directly.
    >>> svc.pack('model', model)
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._model = None

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name + COREML_MODEL_FILE_EXTENTION)

    def pack(self, model, metadata=None):  # pylint:disable=arguments-differ
        try:
            import coremltools
        except ImportError:
            raise MissingDependencyException(
                "coremltools>=4.0b2 package is required to use CoreMLModelArtifact"
            )

        if not isinstance(model, coremltools.models.MLModel):
            raise InvalidArgument(
                "CoreMLModelArtifact can only pack type 'coremltools.models.MLModel'"
            )

        self._model = model
        return self

    def load(self, path):
        try:
            import coremltools
        except ImportError:
            raise MissingDependencyException(
                "coremltools package is required to use CoreMLModelArtifact"
            )

        model = coremltools.models.MLModel(self._file_path(path))

        if not isinstance(model, coremltools.models.MLModel):
            raise InvalidArgument(
                "Expecting CoreMLModelArtifact loaded object type to be "
                "'coremltools.models.MLModel' but actually it is {}".format(type(model))
            )

        return self.pack(model)

    def set_dependencies(self, env: BentoServiceEnv):
        if env._infer_pip_packages:
            env.add_pip_packages(['coremltools>=4.0b2'])

    def get(self):
        return self._model

    def save(self, dst):
        self._model.save(self._file_path(dst))
