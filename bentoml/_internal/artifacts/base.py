import os
import typing as t
from pathlib import Path

from ..types import MetadataType, PathType
from ..utils.ruamel_yaml import YAML

BA = t.TypeVar("BA", bound="ModelArtifact")

MT = t.TypeVar("MT")


class _ArtifactMeta(type):
    """
    Metaclass for all Artifacts. We want to add the
    following function to each class:

    - get_path(cls, path: PathType, ext: str) -> PathType:
        returns path of saved model with its saved type extension.
        This can be used at class level.
        (e.g: model.pkl(pickle), model.pt(torch), model.h5(keras), and so on.)

    This will also add default file extension that most frameworks
    will use as class attributes.
    """

    _MODEL_NAMESPACE: str = "model"
    _FILE_ENCODING: str = 'utf-8'

    _FILE_EXTENSION: t.Dict[str, str] = {
        "H5_EXTENSION": ".h5",
        "HDF5_EXTENSION": ".hdf5",
        "JSON_EXTENSION": ".json",
        "PICKLE_EXTENSION": ".pkl",
        "PTH_EXTENSION": ".pth",
        "PT_EXTENSION": ".pt",
        "TXT_EXTENSION": ".txt",
        "YAML_EXTENSION": ".yaml",
        "YML_EXTENSION": ".yml",
    }

    @classmethod
    def __get__path(
        cls, path: PathType, ext: t.Optional[str] = ""
    ) -> PathType:  # pylint: disable=unused-private-member
        """
        Return a default saved path for implemented artifacts.

        Args:
            path (`Union[str, os.PathLike]`):
                Given path containing saved artifact.
            ext (`str`, `optional`, default to `""`):
                Given extension. Some frameworks doesn't require
                a specified file extension, hence the behaviour
                of empty string.

        Returns:
            :obj:`str` which is the default saved path of given implemented artifacts.

        ::

            PyTorchModel.get_path(os.getcwd(),".pt") # will return os.getcwd()/model.pt
        """
        return os.path.join(path, f"{cls._MODEL_NAMESPACE}{ext}")

    def __new__(mcls, name, mixins, namespace):
        kwargs: t.Dict[str, t.Callable] = dict()
        kwargs.update(
            {
                "_MODEL_NAMESPACE": mcls._MODEL_NAMESPACE,
                "_FILE_ENCODING": mcls._FILE_ENCODING,
                "get_path": mcls.__get__path,
            },
            **mcls._FILE_EXTENSION,
        )
        namespace.update(**kwargs)
        return super(_ArtifactMeta, mcls).__new__(mcls, name, mixins, namespace)


class ModelArtifact(object, metaclass=_ArtifactMeta):
    """
    :class:`~bentoml._internal.artifacts.ModelArtifact` is
    the base abstraction for describing the trained model
    serialization and deserialization process.

    Args:
        model (`MT`):
            Given model definition. Omit various type depending on given frameworks.
        metadata (`Dict[str, Any]`, or :obj:`~bentoml._internal.types.MetadataType`, `optional`, default to `None`):
            Class metadata
    
    We don't want to abstract a lot of framework specific library code when creating new
    BentoML artifacts. This means we prefer duplication of codes for helper function
    rather than bad design abstraction. When create helper function in specific frameworks,
    make sure that those helpers function are reserved, and follow the below format:

    .. note::   
        The reason for doing this is to make the library code more consistent and easier maintainability.
        
    .. code-block:: python
    
        class KerasModel(ModelArtifact):
            
            # function should start with two underscore, followed
            #   by its descriptive function, with two underscore following it
            #   and the desired scope of that function to apply to
            def __helper_func_for__path(path: str, ...):
                # this is a helper function that is related to path
                pass

    .. note:: 
        Make sure to add ``# noqa # pylint: disable=arguments-differ`` to :meth:`load` when implementing 
        newly integration or custom artifacts if the behaviour of ``load`` subclass takes different parameters
        
        .. code-block:: python

            from bentoml._internal.artifacts import ModelArtifact
            
            class CustomArtifact(ModelArtifact):
                def __init__(self, model, metadata=None):...

                @classmethod
                def load(cls, path: str, args1, args2):...  # noqa # pylint: disable=arguments-differ
            
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
        :meth:`~bentoml._internal.artifacts.ModelArtifact.save` to load model during
        development pipeline.
        """  # noqa: E501

    def save(self: BA, path: PathType) -> None:
        """
        Perform save instance to given path.

        Args:
            path (`Union[str, os.PathLike]`, or :obj:`~bentoml._internal.types.PathType`):
                Given path to save artifacts metadata and objects.

        Usually this can be used with :meth:`~bentoml._internal.artifacts.ModelArtifact.load` to load
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
        """  # noqa: E501

    def __getattribute__(self: BA, item: str):
        if item == 'save':

            def wrapped_save(*args, **kw):
                # workaround method overloading.
                path = args[0]  # save(self, path)
                if self.metadata:
                    yaml = YAML()
                    yaml.dump(
                        self.metadata, Path(self.get_path(path, self.YML_EXTENSION)),
                    )

                inherited = object.__getattribute__(self, item)
                return inherited(*args, **kw)

            return wrapped_save
        elif item == 'load':

            def wrapped_load(*args, **kw):
                assert (
                    'path' in args
                ), 'load() implementation requires positional first args `path`'
                inherited = object.__getattribute__(self, item)
                return inherited(*args, **kw)

            return wrapped_load
        else:
            return object.__getattribute__(self, item)
