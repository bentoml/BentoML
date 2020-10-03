import logging
import os

from bentoml.exceptions import InvalidArgument, MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv
from bentoml.utils import cloudpickle

logger = logging.getLogger(__name__)


class PytorchModelArtifact(BentoServiceArtifact):
    """
    Abstraction for saving/loading objects with torch.save and torch.load

    Args:
        name (string): name of the artifact

    Raises:
        MissingDependencyException: torch package is required for PytorchModelArtifact
        InvalidArgument: invalid argument type, model being packed must be instance of
            torch.nn.Module

    Example usage:

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
    >>> # Train model with data
    >>>
    >>>
    >>> import bentoml
    >>> from bentoml.adapters import ImageInput
    >>> from bentoml.frameworks.pytorch import PytorchModelArtifact
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts([PytorchModelArtifact('net')])
    >>> class PytorchModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=ImageInput(), batch=True)
    >>>     def predict(self, imgs):
    >>>         outputs = self.artifacts.net(imgs)
    >>>         return outputs
    >>>
    >>>
    >>> svc = PytorchModelService()
    >>>
    >>> # Pytorch model can be packed directly.
    >>> svc.pack('net', net)
    """

    def __init__(self, name, file_extension=".pt"):
        super(PytorchModelArtifact, self).__init__(name)
        self._file_extension = file_extension
        self._model = None

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name + self._file_extension)

    def pack(self, model):  # pylint:disable=arguments-differ
        try:
            import torch
        except ImportError:
            raise MissingDependencyException(
                "torch package is required to use PytorchModelArtifact"
            )

        if not isinstance(model, torch.nn.Module):
            raise InvalidArgument(
                "PytorchModelArtifact can only pack type 'torch.nn.Module'"
            )

        self._model = model
        return self

    def load(self, path):
        try:
            import torch
        except ImportError:
            raise MissingDependencyException(
                "torch package is required to use PytorchModelArtifact"
            )

        model = cloudpickle.load(open(self._file_path(path), 'rb'))

        if not isinstance(model, torch.nn.Module):
            raise InvalidArgument(
                "Expecting PytorchModelArtifact loaded object type to be "
                "'torch.nn.Module' but actually it is {}".format(type(model))
            )

        return self.pack(model)

    def set_dependencies(self, env: BentoServiceEnv):
        logger.warning(
            "BentoML by default does not include spacy and torchvision package when "
            "using PytorchModelArtifact. To make sure BentoML bundle those packages if "
            "they are required for your model, either import those packages in "
            "BentoService definition file or manually add them via "
            "`@env(pip_packages=['torchvision'])` when defining a BentoService"
        )
        env.add_pip_packages(['torch'])

    def get(self):
        return self._model

    def save(self, dst):
        return cloudpickle.dump(self._model, open(self._file_path(dst), "wb"))
