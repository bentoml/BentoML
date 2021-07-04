import logging
import os
import zipfile
import pathlib
import shutil

from bentoml.exceptions import (
    InvalidArgument,
    MissingDependencyException,
)
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv
from bentoml.utils import cloudpickle

logger = logging.getLogger(__name__)


def _is_path_like(path):
    return isinstance(path, (str, bytes, pathlib.Path, os.PathLike))


def _is_pytorch_lightning_model_file_like(path):
    return (
        _is_path_like(path)
        and os.path.isfile(path)
        and str(path).lower().endswith(".pt")
    )


class PytorchModelArtifact(BentoServiceArtifact):
    """
    Artifact class for saving/loading objects with torch.save and torch.load

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
    >>>
    >>> # Alternatively,
    >>>
    >>> # Pack a TorchScript Model
    >>> # Random input in the format expected by the net
    >>> sample_input = ...
    >>> traced_net = torch.jit.trace(net, sample_input)
    >>> svc.pack('net', traced_net)
    """

    def __init__(self, name, file_extension=".pt"):
        super().__init__(name)
        self._file_extension = file_extension
        self._model = None

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name + self._file_extension)

    def pack(self, model, metadata=None):  # pylint:disable=arguments-renamed
        try:
            import torch
        except ImportError:
            raise MissingDependencyException(
                "torch package is required to use PytorchModelArtifact"
            )

        if not isinstance(model, torch.nn.Module):
            raise InvalidArgument(
                "PytorchModelArtifact can only pack type \
                'torch.nn.Module' or 'torch.jit.ScriptModule'"
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

        # TorchScript Models are saved as zip files
        if zipfile.is_zipfile(self._file_path(path)):
            model = torch.jit.load(self._file_path(path))
        else:
            model = cloudpickle.load(open(self._file_path(path), 'rb'))

        if not isinstance(model, torch.nn.Module):
            raise InvalidArgument(
                "Expecting PytorchModelArtifact loaded object type to be "
                "'torch.nn.Module' or 'torch.jit.ScriptModule' \
                but actually it is {}".format(
                    type(model)
                )
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
        if env._infer_pip_packages:
            env.add_pip_packages(['torch'])

    def get(self):
        return self._model

    def save(self, dst):
        try:
            import torch
        except ImportError:
            raise MissingDependencyException(
                "torch package is required to use PytorchModelArtifact"
            )

        # If model is a TorchScriptModule, we cannot apply standard pickling
        if isinstance(self._model, torch.jit.ScriptModule):
            return torch.jit.save(self._model, self._file_path(dst))

        return cloudpickle.dump(self._model, open(self._file_path(dst), "wb"))


class PytorchLightningModelArtifact(BentoServiceArtifact):
    """
    Artifact class for saving and loading pytorch lightning model

    Args:
        name (string): Name of the pytorch model
    Raises:
        MissingDependencyException: torch and pytorch_lightning package is required.

    Example usage:

    >>>
    >>> # Train pytorch lightning model
    >>> from pytorch_lightning.core.lightning import LightningModule
    >>>
    >>> class SimpleModel(LightningModule):
    >>>     def forward(self, x):
    >>>         return x.add(1)
    >>>
    >>> model = SimpleModel()
    >>>
    >>> import bentoml
    >>> import torch
    >>> from bentoml.adapters import JsonInput
    >>> from bentoml.frameworks.pytorch import PytorchLightningModelArtifact
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts([PytorchLightningModelArtifact('model')])
    >>> class PytorchLightingService(bentoml.BentoService):
    >>>     @bentoml.api(input=DataframeInput(), batch=True)
    >>>     def predict(self, df):
    >>>         input_tensor = torch.from_numpy(df.to_numpy())
    >>>         return self.artifacts.model(input).numpy()
    >>>
    >>> svc = PytorchLightingService()
    >>> # Pack Pytorch Lightning model instance
    >>> svc.pack('model', model)
    >>>
    >>> # Pack saved Pytorch Lightning model
    >>> # import torch
    >>> # script = model.to_torchscript()
    >>> # saved_model_path = 'path/to/model.pt'
    >>> # torch.jit.save(script, saved_model_path)
    >>> # svc.pack('model', saved_model_path)
    >>>
    >>> svc.save()
    """

    def __init__(self, name):
        super().__init__(name)
        self._model = None
        self._model_path = None

    def _saved_model_file_path(self, base_path):
        return os.path.join(base_path, self.name + '.pt')

    def pack(self, path_or_model, metadata=None):  # pylint:disable=arguments-renamed
        if _is_pytorch_lightning_model_file_like(path_or_model):
            logger.info(
                'PytorchLightningArtifact is packing a saved torchscript module '
                'from path'
            )
            self._model_path = path_or_model
        else:
            try:
                from pytorch_lightning.core.lightning import LightningModule
            except ImportError:
                raise InvalidArgument(
                    '"pytorch_lightning.lightning.LightningModule" model is required '
                    'to pack a PytorchLightningModelArtifact'
                )
            if isinstance(path_or_model, LightningModule):
                logger.info(
                    'PytorchLightningArtifact is packing a pytorch lightning '
                    'model instance as torchscript module'
                )
                self._model = path_or_model.to_torchscript()
            else:
                raise InvalidArgument(
                    'a LightningModule model is required to pack a '
                    'PytorchLightningModelArtifact'
                )
        return self

    def load(self, path):
        self._model = self._get_torch_script_model(self._saved_model_file_path(path))

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_packages(['pytorch-lightning'])

    def get(self):
        if self._model is None:
            self._model = self._get_torch_script_model(self._model_path)
        return self._model

    def save(self, dst):
        if self._model:
            try:
                import torch
            except ImportError:
                raise MissingDependencyException(
                    '"torch" package is required for saving Pytorch lightning model'
                )
            torch.jit.save(self._model, self._saved_model_file_path(dst))
        if self._model_path:
            shutil.copyfile(self._model_path, self._saved_model_file_path(dst))

    @staticmethod
    def _get_torch_script_model(model_path):
        try:
            from torch import jit
        except ImportError:
            raise MissingDependencyException(
                '"torch" package is required for inference with '
                'PytorchLightningModelArtifact'
            )
        return jit.load(model_path)
