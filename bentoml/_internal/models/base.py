import typing as t
from pathlib import Path

import yaml

from ..types import MetadataType, PathType

MT = t.TypeVar("MT", bound=t.Any)

H5_EXTENSION: str = ".h5"
HDF5_EXTENSION: str = ".hdf5"
JSON_EXTENSION: str = ".json"
PICKLE_EXTENSION: str = ".pkl"
PTH_EXTENSION: str = ".pth"
PT_EXTENSION: str = ".pt"
TXT_EXTENSION: str = ".txt"
YAML_EXTENSION: str = ".yaml"
MODEL_NAMESPACE: str = "bentoml_model"


def _validate_or_create_dir(path: PathType) -> None:
    path = Path(path)

    if path.exists():
        if not path.is_dir():
            raise OSError(20, f"{path} is not a directory")
    else:
        path.mkdir(parents=True)


class Model(object):
    """
    :class:`Model` is the base abstraction
     for describing the trained model serialization
     and deserialization process.

    Args:
        model (`MT`):
            Given model definition. Omit various type depending on given frameworks.
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    .. note::
        Make sure to add ``# noqa # pylint: disable=arguments-differ`` to :meth:`load` when implementing
        newly integration or custom artifacts if the behaviour of ``load`` subclass takes different parameters

        .. code-block:: python

            from bentoml._internal.artifacts import Model

            class CustomModel(Model):
                def __init__(self, model, metadata=None):...

                @classmethod
                def load(cls, path: str, args1, args2):...  # noqa # pylint: disable=arguments-differ

    Example usage for creating a custom ``Model``::

        TODO:
    """

    def __init__(self: "Model", model: MT, metadata: t.Optional[MetadataType] = None):
        self._model = model
        self._metadata = metadata

    @property
    def metadata(self: "Model") -> t.Optional[MetadataType]:
        return self._metadata

    @classmethod
    def load(cls, path: PathType, **kwargs) -> t.Any:
        """
        Load saved model into memory.

        Args:
            path (`Union[str, os.PathLike]`):
                Given path to save artifacts metadata and objects.

        This will be used as a class method, interchangeable with
        :meth:`save` to load model during development pipeline.
        """
        raise NotImplementedError

    def save(self: "Model", path: PathType) -> None:
        """
        Perform save instance to given path.

        Args:
            path (`Union[str, os.PathLike]`, or :obj:`~bentoml._internal.types.PathType`):
                Given path to save artifacts metadata and objects.

        Usually this can be used with :meth:`~bentoml._internal.artifacts.Model.load` to load
        model objects for development::

            # train.py
            model = MyPyTorchModel().train()  # type: torch.nn.Module
            ...
            from bentoml.pytorch import PyTorchModel
            PyTorchModel(model).save(".")
            pytorch_model = PyTorchModel.load(".")  # type: torch.nn.Module

        .. admonition:: current implementation

            Current implementation initialize base :meth:`save()` and :meth:`load()` in
            :code:`__getattribute__()` via wrapper. Since Python doesn't have support
            for method overloading, this ensures that model metadata will always be saved
            to given directory.
        """  # noqa # pylint: enable=line-too-long
        raise NotImplementedError

    def __getattribute__(self: "Model", item: str) -> t.Any:
        if item == "save":

            def wrapped_save(*args, **kw):  # type: ignore
                path: PathType = args[0]  # save(self, path)
                _validate_or_create_dir(path)
                if self.metadata:
                    with open(
                        Path(path, f"{MODEL_NAMESPACE}{YAML_EXTENSION}", "r")
                    ) as f:
                        yaml.dump(self.metadata, f)

                inherited = object.__getattribute__(self, item)
                return inherited(*args, **kw)

            return wrapped_save
        elif item == "load":

            def wrapped_load(*args, **kw):  # type: ignore
                assert_msg: str = "`load()` requires positional `path`"
                assert "path" in args, assert_msg
                inherited = object.__getattribute__(self, item)
                return inherited(*args, **kw)

            return wrapped_load
        else:
            return object.__getattribute__(self, item)
