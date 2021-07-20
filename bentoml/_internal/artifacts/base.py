import os
import typing as t
from pathlib import Path

from ..types import MetadataType, PathType
from ..utils.ruamel_yaml import YAML

BA = t.TypeVar("BA", bound="BaseArtifact")

MT = t.TypeVar("MT")


_FILE_EXTENSION: t.Dict[str, str] = {
    "PICKLE_FILE_EXTENSION": ".pkl",
    "TXT_FILE_EXTENSION": ".txt",
    "JSON_FILE_EXTENSION": ".json",
    "YML_FILE_EXTENSION": ".yml",
    "YAML_FILE_EXTENSION": ".yaml",
    "PT_FILE_EXTENSION": ".pt",
    "PTH_FILE_EXTENSION": ".pth",
    "H5_FILE_EXTENSION": ".h5",
    "HDF5_FILE_EXTENSION": ".hdf5",
}


class ArtifactMeta(type):
    """
    Metaclass for all Artifacts. We want to add the
    following function to each class:

    - model_path(cls, path: PathType, ext: str) -> PathType:
        returns path of saved model with its saved type extension.
        This can be used at class level.
        (e.g: model.pkl(pickle), model.pt(torch), model.h5(keras), and so on.)
    - walk_path(cls, path: PathType, ext: str) -> pathlib.Path:
        Iterate through given path and returns a :obj:`pathlib.Path` object.
        Some frameworks (mlflow, h2o) incorporates the concept of projects within
        their saving artifacts, thus we want a way to iterate through those directory.

    This will also add default file extension that most frameworks
    will use as class attributes.
    """

    _MODEL_NAMESPACE: str = "model"

    @classmethod
    def __model_path(cls, path: PathType, ext: str) -> PathType:
        return os.path.join(path, f"{cls._MODEL_NAMESPACE}{ext}")

    @classmethod
    def __walk_path(cls, path: PathType, ext: str) -> Path:
        try:
            for f in Path(path).iterdir():
                if f.is_dir():
                    return cls.__walk_path(str(f), ext)
                elif f.suffix == ext:
                    return f
                else:
                    continue
        except FileNotFoundError:
            raise

    def __new__(cls, name, mixins, namespace):
        _path_fn: t.Dict[str, t.Callable[[PathType, str], t.Union[PathType, Path]]] = {
            "model_path": cls.__model_path,
            "walk_path": cls.__walk_path,
        }
        _FILE_EXTENSION.update({"_MODEL_NAMESPACE": cls._MODEL_NAMESPACE})
        namespace.update(_path_fn)
        for k, v in _FILE_EXTENSION.items():
            namespace[k] = v
        return type.__new__(cls, name, mixins, namespace)


class BaseArtifact(object, metaclass=ArtifactMeta):
    """
    :class:`~bentoml._internal.artifacts.BaseModelArtifact` is
    the base abstraction for describing the trained model
    serialization and deserialization process.

    Args:
        model (`MT`):
            Given model definition. Omit various type depending on given frameworks.
        metadata (`Dict[str, Any]`, or :obj:`~bentoml._internal.types.MetadataType`, `optional`, default to `None`):
            Class metadata

    Example usage for creating a custom ``ModelArtifacts``::

        TODO:
    """  # noqa: E501

    def __init__(self: BA, model: MT, metadata: t.Optional[MetadataType] = None):
        self._model = model
        self._metadata = metadata

    @property
    def metadata(self: BA) -> MetadataType:
        return self._metadata

    @classmethod
    def load(cls: BA, path: PathType) -> MT:
        """
        Load saved model into memory.

        Args:
            path (`Union[str, os.PathLike]`, or :obj:`~bentoml._internal.types.PathType`):
                Given path to save artifacts metadata and objects.

        This will be used as a class method, interchangeable with
        :meth:`~bentoml._internal.artifacts.BaseArtifact.save` to load model during
        development pipeline.
        """  # noqa: E501

        inherited = object.__getattribute__(cls, "load")
        return inherited(path)

    def save(self: BA, path: PathType) -> None:
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
            PyTorchModel(model).save(".")
            pytorch_model = PyTorchModel.load(".")  # type: torch.nn.Module

        .. admonition:: current implementation

            Current implementation initialize base :meth:`~bentoml._internal.artifacts.BaseArtifact.save`
            in :code:`__getattribute__` via wrapper. Since Python doesn't have support for method overloading,
            This ensures that model metadata will always be saved to given directory.
        """  # noqa: E501

    def __getattribute__(self: BA, item: str):
        if item == 'save':

            def wrapped_save(*args, **kw):
                # workaround method overloading.
                path = args[0]  # save(self, path)
                if self.metadata:
                    yaml = YAML()
                    yaml.dump(
                        self.metadata,
                        Path(self.model_path(path, self.YML_FILE_EXTENSION)),
                    )

                inherited = object.__getattribute__(self, item)
                return inherited(*args, **kw)

            return wrapped_save

        return object.__getattribute__(self, item)
