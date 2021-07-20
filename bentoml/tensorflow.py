# ==============================================================================
#     Copyright (c) 2021 Atalaya Tech. Inc
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================

import logging
import pathlib
import shutil
import tempfile
import importlib
import os
import typing as t

from ._internal.artifacts import BaseArtifact

from ._internal.utils.tensorflow import (
    cast_tensor_by_spec,
    get_arg_names,
    get_input_signatures,
    get_restored_functions,
    pretty_format_restored_model,
)
from ._internal.exceptions import (
    ArtifactLoadingException,
    InvalidArgument,
    MissingDependencyException,
)
from ._internal.types import MetadataType, PathType
from ._internal.utils import cloudpickle

try:
    import tensorflow as tf
except ImportError:
    raise MissingDependencyException("tensorflow is required by TensorflowModel.")

try:
    import tensorflow.keras  # pylint: disable=unused-import
except ImportError:
    raise MissingDependencyException("tensorflow is required by KerasModel as backend runtime.")


MODULE_NAME_FILE_ENCODING: str = "utf-8"

KT = t.TypeVar("KT", bound=tf.keras.models.Model)


from ._internal.artifacts import BaseArtifact

logger = logging.getLogger(__name__)


def _is_path_like(p):
    return isinstance(p, (str, bytes, pathlib.PurePath, os.PathLike))


class _TensorflowFunctionWrapper:
    """
    TensorflowFunctionWrapper
    transform input tensor following function input signature
    """

    def __init__(self, origin_func, arg_names=None, arg_specs=None, kwarg_specs=None):
        self.origin_func = origin_func
        self.arg_names = arg_names
        self.arg_specs = arg_specs
        self.kwarg_specs = {k: v for k, v in zip(arg_names or [], arg_specs or [])}
        self.kwarg_specs.update(kwarg_specs or {})

    def __call__(self, *args, **kwargs):
        if self.arg_specs is None and self.kwarg_specs is None:
            return self.origin_func(*args, **kwargs)

        for k in kwargs:
            if k not in self.kwarg_specs:
                raise TypeError(f"Function got an unexpected keyword argument {k}")

        arg_keys = {k for k, _ in zip(self.arg_names, args)}
        _ambiguous_keys = arg_keys & set(kwargs)
        if _ambiguous_keys:
            raise TypeError(f"got two values for arguments '{_ambiguous_keys}'")

        # INFO:
        # how signature with kwargs works?
        # https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/eager/function.py#L1519

        transformed_args = tuple(
            cast_tensor_by_spec(arg, spec) for arg, spec in zip(args, self.arg_specs)
        )

        transformed_kwargs = {
            k: cast_tensor_by_spec(arg, self.kwarg_specs[k])
            for k, arg in kwargs.items()
        }
        return self.origin_func(*transformed_args, **transformed_kwargs)

    def __getattr__(self, k):
        return getattr(self.origin_func, k)

    @classmethod
    def hook_loaded_model(cls, loaded_model):
        funcs = get_restored_functions(loaded_model)
        for k, func in funcs.items():
            arg_names = get_arg_names(func)
            sigs = get_input_signatures(func)
            if not sigs:
                continue
            arg_specs, kwarg_specs = sigs[0]
            setattr(
                loaded_model,
                k,
                cls(
                    func,
                    arg_names=arg_names,
                    arg_specs=arg_specs,
                    kwarg_specs=kwarg_specs,
                ),
            )


def _load_tf_saved_model(path):
    try:
        import tensorflow as tf
        from tensorflow.python.training.tracking.tracking import AutoTrackable

        TF2 = tf.__version__.startswith("2")
    except ImportError:
        raise MissingDependencyException(
            "Tensorflow package is required to use TfSavedModelArtifact"
        )

    if TF2:
        return tf.saved_model.load(path)
    else:
        loaded = tf.compat.v2.saved_model.load(path)
        if isinstance(loaded, AutoTrackable) and not hasattr(loaded, "__call__"):
            logger.warning(
                """Importing SavedModels from TensorFlow 1.x.
                `outputs = imported(inputs)` is not supported in bento service due to
                tensorflow API.

                Recommended usage:

                ```python
                from tensorflow.python.saved_model import signature_constants

                imported = tf.saved_model.load(path_to_v1_saved_model)
                wrapped_function = imported.signatures[
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
                wrapped_function(tf.ones([]))
                ```

                See https://www.tensorflow.org/api_docs/python/tf/saved_model/load for
                details.
                """
            )
        return loaded


class TensorflowSavedModelArtifact(BaseArtifact):
    """
    Artifact class for saving/loading Tensorflow model in tf.saved_model format

    Args:
        name (string): name of the artifact

    Raises:
        MissingDependencyException: tensorflow package is required for
            TensorflowSavedModelArtifact

    Example usage:

    >>> import tensorflow as tf
    >>>
    >>> # Option 1: custom model with specific method call
    >>> class Adder(tf.Module):
    >>>     @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    >>>     def add(self, x):
    >>>         return x + x + 1.
    >>> model_to_save = Adder()
    >>> # ... compiling, training, etc
    >>>
    >>> # Option 2: Sequential model (direct call only)
    >>> model_to_save = tf.keras.Sequential([
    >>>     tf.keras.layers.Flatten(input_shape=(28, 28)),
    >>>     tf.keras.layers.Dense(128, activation='relu'),
    >>>     tf.keras.layers.Dense(10, activation='softmax')
    >>> ])
    >>> # ... compiling, training, etc
    >>>
    >>> import bentoml
    >>> from bentoml.adapters import JsonInput
    >>> from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
    >>>
    >>> @bentoml.env(pip_packages=["tensorflow"])
    >>> @bentoml.artifacts([TensorflowSavedModelArtifact('model')])
    >>> class TfModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=JsonInput(), batch=False)
    >>>     def predict(self, json):
    >>>         input_data = json['input']
    >>>         prediction = self.artifacts.model.add(input_data)
    >>>         # prediction = self.artifacts.model(input_data)  # if Sequential mode
    >>>         return prediction.numpy()
    >>>
    >>> svc = TfModelService()
    >>>
    >>> # Option 1: pack directly with Tensorflow trackable object
    >>> svc.pack('model', model_to_save)
    >>>
    >>> # Option 2: save to file path then pack
    >>> tf.saved_model.save(model_to_save, '/tmp/adder/1')
    >>> svc.pack('model', '/tmp/adder/1')
    """

    def __init__(self, name):
        super().__init__(name)

        self._model = None
        self._tmpdir = None
        self._path = None

    def _saved_model_path(self, base_path):
        return os.path.join(base_path, self.name + "_saved_model")

    def pack(
        self, obj, metadata=None, signatures=None, options=None
    ):  # pylint:disable=arguments-differ
        """
        Pack the TensorFlow Trackable object `obj` to [SavedModel format].
        Args:
          obj: A trackable object to export.
          signatures: Optional, either a `tf.function` with an input signature
            specified or the result of `f.get_concrete_function` on a
            `@tf.function`-decorated function `f`, in which case `f` will be used to
            generate a signature for the SavedModel under the default serving
            signature key. `signatures` may also be a dictionary, in which case it
            maps from signature keys to either `tf.function` instances with input
            signatures or concrete functions. The keys of such a dictionary may be
            arbitrary strings, but will typically be from the
            `tf.saved_model.signature_constants` module.
          options: Optional, `tf.saved_model.SaveOptions` object that specifies
            options for saving.

        Raises:
          ValueError: If `obj` is not trackable.
        """
        if not _is_path_like(obj):
            if self._tmpdir is not None:
                self._tmpdir.cleanup()
            else:
                self._tmpdir = tempfile.TemporaryDirectory()
            try:
                import tensorflow as tf

                TF2 = tf.__version__.startswith("2")
            except ImportError:
                raise MissingDependencyException(
                    "Tensorflow package is required to use TfSavedModelArtifact."
                )
            if TF2:
                tf.saved_model.save(
                    obj, self._tmpdir.name, signatures=signatures, options=options,
                )
            else:
                if self.options:
                    logger.warning(
                        "Parameter 'options: %s' is ignored when using Tensorflow "
                        "version 1",
                        str(options),
                    )
                tf.saved_model.save(
                    obj, self._tmpdir.name, signatures=signatures,
                )
            self._path = self._tmpdir.name
        else:
            self._path = obj
        self._packed = True
        loaded = self.get()
        logger.warning(
            "Due to TensorFlow's internal mechanism, only methods wrapped under "
            "`@tf.function` decorator and the Keras default function "
            "`__call__(inputs, training=False)` can be restored after a save & load.\n"
            "You can test the restored model object by referring:\n"
            f"<bento_svc>.artifacts.{self.name}\n"
        )
        logger.info(pretty_format_restored_model(loaded))
        if hasattr(loaded, "keras_api"):
            logger.warning(
                f"BentoML detected that {self.__class__.__name__} is being used "
                "to pack a Keras API based model. "
                "In order to get optimal serving performance, we recommend "
                f"either replacing {self.__class__.__name__} with KerasModel, "
                "or wrapping the keras_model.predict method with tf.function decorator."
            )
        return self

    def get(self):
        if self._model is None:
            loaded_model = _load_tf_saved_model(self._path)
            _TensorflowFunctionWrapper.hook_loaded_model(loaded_model)
            self._model = loaded_model
        return self._model

    def load(self, path):
        saved_model_path = self._saved_model_path(path)
        return self.pack(saved_model_path)

    def save(self, dst):
        # Copy exported SavedModel model directory to BentoML saved artifact directory
        shutil.copytree(self._path, self._saved_model_path(dst))

    def __del__(self):
        if getattr(self, "_tmpdir", None) is not None:
            self._tmpdir.cleanup()

class KerasModel(BaseArtifact):
    """
    Model class for saving/loading :obj:`keras` models using Tensorflow backend.

    Args:
        model (`tf.keras.models.Model`):
            Keras model instance and its subclasses.
        default_custom_objects (`Dict[str, Any]`, `optional`, default to `None`):
            dictionary of Keras custom objects for model
        store_as_json_and_weights (`bool`, `optional`, default to `False`):
            flag allowing storage of the Keras model as JSON and weights
        metadata (`Dict[str, Any]`, or :obj:`~bentoml._internal.types.MetadataType`, `optional`, default to `None`):
            Class metadata
        name (`str`, `optional`, default to `kerasmodel`):
            KerasModel instance name

    Raises:
        MissingDependencyException:
            :obj:`tensorflow` package is required for KerasModel
        InvalidArgument:
            model being packed must be instance of :class:`keras.engine.network.Network`,
            :class:`tf.keras.models.Model`, or its subclasses

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """  # noqa: E501

    def __init__(
            self,
            model: KT,
            default_custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
            store_as_json_and_weights: t.Optional[bool] = False,
            metadata: t.Optional[MetadataType] = None,
            name: t.Optional[str] = "kerasmodel",
    ):
        super(KerasModel, self).__init__(model, metadata=metadata, name=name)

        self._default_custom_objects = default_custom_objects
        self._store_as_json_and_weights = store_as_json_and_weights

        # By default assume using tf.keras module
        self._keras_module_name = tf.keras.__name__
        self.sess = None
        self.graph = None

        self._custom_objects = None

    def _custom_objects_path(self, path: PathType) -> PathType:
        return self.model_path(path, f"_custom_objects{self.PICKLE_FILE_EXTENSION}")

    def _model_file_path(self, path: PathType) -> PathType:
        return self.model_path(path, self.H5_FILE_EXTENSION)

    def _model_weights_path(self, path: PathType) -> PathType:
        return self.model_path(path, f"_weights{self.HDF5_FILE_EXTENSION}")

    def _model_json_path(self, path: PathType) -> PathType:
        return self.model_path(path, f"_json{self.JSON_FILE_EXTENSION}")

    def bind_keras_backend_session(self):
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.graph = self.sess.graph

    def create_session(self):
        self.graph = tf.compat.v1.get_default_graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        tf.compat.v1.keras.backend.set_session(self.sess)

    def pack(self, data):
        if isinstance(data, dict):
            model = data["model"]
            custom_objects = (
                data["custom_objects"]
                if "custom_objects" in data
                else self._default_custom_objects
            )
        else:
            model = data
            custom_objects = self._default_custom_objects

        if not isinstance(model, tf.keras.models.Model):
            error_msg = rf"""\
                Expects model argument of type `tf.keras.models.Model`,
                got type: {type(model)} instead
            """
            raise InvalidArgument(error_msg)

        self._model = model
        self._custom_objects = custom_objects
        return self

    def load(self, path):
        if os.path.isfile(self._keras_module_name_path(path)):
            with open(self._keras_module_name_path(path), "rb") as text_file:
                keras_module_name = text_file.read().decode(MODULE_NAME_FILE_ENCODING)
                try:
                    keras_module = importlib.import_module(keras_module_name)
                except ImportError:
                    raise ArtifactLoadingException(
                        f"Failed to import '{keras_module_name}' module when"
                        "loading saved KerasModel"
                    )
        else:
            raise ArtifactLoadingException(
                "Failed to read keras model name from '{}' when loading saved "
                "KerasModel".format(self._keras_module_name_path(path))
            )

        if self._default_custom_objects is None and os.path.isfile(
                self._custom_objects_path(path)
        ):
            self._default_custom_objects = cloudpickle.load(
                open(self._custom_objects_path(path), "rb")
            )

        if self._store_as_json_and_weights:
            # load keras model via json and weights if store_as_json_and_weights=True
            self.create_session()
            with self.graph.as_default():
                with self.sess.as_default():
                    with open(self._model_json_path(path), "r") as json_file:
                        model_json = json_file.read()
                    model = keras_module.models.model_from_json(
                        model_json, custom_objects=self._default_custom_objects
                    )
                    model.load_weights(self._model_weights_path(path))
        else:
            # otherwise, load keras model via standard load_model
            model = keras_module.models.load_model(
                self._model_file_path(path),
                custom_objects=self._default_custom_objects,
            )
        return model

    def save(self, path: PathType) -> None:
        # save the keras module name to be used when loading
        with open(self._keras_module_name_path(path), "wb") as text_file:
            text_file.write(self._keras_module_name.encode(MODULE_NAME_FILE_ENCODING))

        # save custom_objects for model
        cloudpickle.dump(
            self._custom_objects, open(self._custom_objects_path(path), "wb")
        )

        if self._store_as_json_and_weights:
            # save keras model using json and weights if requested
            with open(self._model_json_path(path), "w") as json_file:
                json_file.write(self._model.to_json())
            self._model.save_weights(self._model_weights_path(path))
        else:
            # otherwise, save standard keras model
            self._model.save(self._model_file_path(path))