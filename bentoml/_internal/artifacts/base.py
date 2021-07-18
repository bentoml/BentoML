import os
import typing as t
from pathlib import Path

from ..types import MT, PathType
from ..utils.ruamel_yaml import YAML


class ArtifactMeta(type):
    """
    Metaclass for all Artifacts. We want to add the
    following function to each class:

    -model_path(cls, path: PathType, ext: str) -> PathType:
        returns path of saved model with its saved type extension.
        This can be used at class level.
        (e.g: .pkl(pickle), .pt(torch), .h5(keras), and so on.)
    - get_path(path: PathType, ext: str) -> pathlib.Path:
        similar to :meth:`~model_path`, but can be accessed
        as a staticmethod for using at instance level.

    .. note::
        Implement :code:`__init__` when refactoring if needed.
    """

    def _path__model_(cls, path: PathType, ext: str) -> PathType:
        try:
            return PathType(os.path.join(path, getattr(cls, "_model").__name__ + ext))
        except AttributeError:
            # some model class don't have __name__ attributes, then we use default
            # BentoML model namespace
            return PathType(os.path.join(path, cls.__name__ + ext))

    @staticmethod
    def _path__get_(path: PathType, ext: str) -> Path:
        try:
            for f in Path(path).iterdir():
                if f.suffix == ext:
                    return f
        except FileNotFoundError:
            raise

    def __new__(cls, name, mixins, namespace):
        if 'model_path' not in namespace:
            namespace['model_path'] = cls._path__model_
        if 'get_path' not in namespace:
            namespace['get_path'] = cls._path__get_
        return super(ArtifactMeta, cls).__new__(cls, name, mixins, namespace)


class BaseArtifact(metaclass=ArtifactMeta):
    """
    :class:`~bentoml._internal.artifacts.BaseModelArtifact` is the base abstraction
    for describing the trained model serialization and deserialization process.
    :class:`~bentoml._internal.artifacts.BaseModelArtifact` is a singleton

    Class attributes:

    - model (`torch.nn.Module`, `tf.keras.Model`, `sklearn.svm.SVC` and many more):
        Given model definition. Can omit various type depending on given frameworks.
    - metadata (`Dict[str, Union[Any,...]]`, `optional`):
        Dictionary of model metadata

    Example of custom Artifacts::

        TODO:
    """

    YML_EXTENSION = ".yml"

    def __init__(self, model: MT, metadata: t.Optional[t.Dict[str, t.Any]] = None):
        self._model = model
        self._metadata = metadata

    @property
    def metadata(self):
        return self._metadata

    @classmethod
    def load(cls, path: PathType) -> MT:
        """
        Load saved model into memory.

        Args:
            path (`Union[str, os.PathLike]`, or :obj:`~bentoml._internal.types.PathType`):
                Given path to save artifacts metadata and objects.

        This will be used as a class method, interchangeable with
        :meth:`~bentoml._internal.artifacts.BaseArtifact.save` to load model during
        development pipeline.
        """

        inherited = object.__getattribute__(cls, "load")
        return inherited(path)

    def save(self, path: PathType):
        """
        Perform save instance to given path.

        Args:
            path (`Union[str, os.PathLike]`, or :obj:`~bentoml._internal.types.PathType`):
                Given path to save artifacts metadata and objects.

        Usually this can be used with :meth:`~bentoml._internal.artifacts.BaseArtifact.load` to load
        model objects for development::

            # train.py
            model = MyPyTorchModel().train()  # type: torch.nn.Module
            ...
            from bentoml.pytorch import PyTorchModel
            pytorch_artifact = PyTorchModel(model).save(".")
            pytorch_model = PyTorchModel.load(".")  # type: torch.nn.Module
 
        .. admonition:: current implementation
            
            Current implementation initialize base :meth:`~bentoml._internal.artifacts.BaseArtifact.save`
            in :code:`__getattribute__` to overcome method overloading by providing a wrapper. This
            ensures that model metadata will always be saved to given directory and won't be overwritten
            by child inheritance.
        """  # noqa: E501

    def __getattribute__(self, item):
        if item == 'save':

            def wrapped_save(*args, **kw):
                # workaround method overloading.
                path = args[0]  # save(self, path)
                if self.metadata:
                    yaml = YAML()
                    yaml.dump(
                        self.metadata, Path(self.model_path(path, self.YML_EXTENSION))
                    )

                inherited = object.__getattribute__(self, item)
                return inherited(*args, **kw)

            return wrapped_save

        return object.__getattribute__(self, item)
