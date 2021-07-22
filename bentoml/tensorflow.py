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
import os
import pathlib
import shutil
import tempfile
import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import MissingDependencyException
from ._internal.types import MetadataType, PathType
from ._internal.utils.tensorflow import (
    cast_tensor_by_spec,
    get_arg_names,
    get_input_signatures,
    get_restored_functions,
    pretty_format_restored_model,
)

logger = logging.getLogger(__name__)

TF1_AUTOTRACKABLE_NOT_CALLABLE_WARNING: str = """\
Importing SavedModels from TensorFlow 1.x. `outputs = imported(inputs)`
is not supported by BentoML due to `tensorflow` API.

Recommended usage::

    ```python
    from tensorflow.python.saved_model import signature_constants

    imported = tf.saved_model.load("path_to_v1_saved_model")
    wrapped_function = imported.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    wrapped_function(tf.ones([]))
    ```

See https://www.tensorflow.org/api_docs/python/tf/saved_model/load for
details.
"""

TF_FUNCTION_WARNING: str = """\
Due to TensorFlow's internal mechanism, only methods
 wrapped under `@tf.function` decorator and the Keras default function
 `__call__(inputs, training=False)` can be restored after a save & load.\n
You can test the restored model object via `TensorflowModel.load(path)`
"""

KERAS_MODEL_WARNING: str = """\
BentoML detected that {name} is being used  to pack a Keras API
 based model. In order to get optimal serving performance, we recommend
 either replacing {name} with KerasModel, or wrapping `keras_model.predict()`
 method with tf.function decorator.
"""


def _path_like(p) -> bool:
    return isinstance(p, (str, bytes, pathlib.PurePath, os.PathLike))


class _TensorflowFunctionWrapper:
    def __init__(
        self,
        origin_func,
        arg_names: t.Optional[str] = None,
        arg_specs=None,
        kwarg_specs=None,
    ):
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


_TensorflowFunctionWrapper.__doc__ = """\
    TODO:
"""


class TensorflowModel(ModelArtifact):
    """
    Artifact class for saving/loading :obj:`tensorflow model
    with :obj:`tensorflow.saved_model` format

    Args:
        model (`Union[tf.keras.Models, tf.Module, PathType, pathlib.PurePath]`):
            Omit every tensorflow model instance of type :obj:`tf.keras.Models` or
            :obj:`tf.Module`
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata


    Raises:
        MissingDependencyException:
            :obj:`tensorflow` is required by TensorflowModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    try:
        import tensorflow as tf
        from tensorflow.python.training.tracking.tracking import AutoTrackable
    except ImportError:
        raise MissingDependencyException("tensorflow is required by TensorflowModel")

    def __init__(
        self,
        model: t.Union[tf.keras.Model, tf.Module, PathType, pathlib.PurePath],
        metadata: t.Optional[MetadataType] = None,
    ):
        super(TensorflowModel, self).__init__(model, metadata=metadata)
        self._tmpdir = tempfile.TemporaryDirectory()

    @staticmethod
    def __load_tf_saved__model(path: str) -> t.Union[AutoTrackable, t.Any]:
        try:
            import tensorflow as tf
            from tensorflow.python.training.tracking.tracking import AutoTrackable

            TF2 = tf.__version__.startswith("2")
        except ImportError:
            raise MissingDependencyException(
                "tensorflow is required by TensorflowModel"
            )
        if TF2:
            return tf.saved_model.load(path)
        else:
            loaded = tf.compat.v2.saved_model.load(path)
            if isinstance(loaded, AutoTrackable) and not hasattr(loaded, "__call__"):
                logger.warning(TF1_AUTOTRACKABLE_NOT_CALLABLE_WARNING)
            return loaded

    @classmethod
    def load(cls, path: PathType):
        model = cls.__load_tf_saved__model(str(path))
        _TensorflowFunctionWrapper.hook_loaded_model(model)
        logger.warning(TF_FUNCTION_WARNING)
        # pretty format loaded model
        logger.info(pretty_format_restored_model(model))
        if hasattr(model, "keras_api"):
            logger.warning(KERAS_MODEL_WARNING.format(name=cls.__name__))
        return model

    def save(
        self,
        path: PathType,
        signatures=None,
        options: t.Optional["tf.saved_model.SaveOptions"] = None,
    ) -> None:
        """
        Save TensorFlow Trackable object `obj` from [SavedModel format] to path.

        Args:
            path (`Union[str, os.PathLike]`):
                Path containing a trackable object to export.
            signatures (`optional`, default to `None`):
              `signatures` is one of three types:

              a `tf.function` with an input signature specified, which will use the default serving signature key

              `f.get_concrete_function` on a `@tf.function` decorated function `f`, in which case f will be used to
                generate a signature for the SavedModel under the default serving signature key,

                  Examples::

                    >>> class Adder(tf.Module):
                    ...   @tf.function
                    ...   def add(self, x):
                    ...     return x + x

                    >>> model = Adder()
                    >>> tf.saved_model.save(
                    ...   model, '/tmp/adder',signatures=model.add.get_concrete_function(
                    ...     tf.TensorSpec([], tf.float32)))

              a dictionary, which maps signature keys to either :obj`tf.function`
                instances with input signatures or concrete functions. Keys of such
                a dictionary may be arbitrary strings, but will typically be from
                the :obj:`tf.saved_model.signature_constants` module.
            options (`tf.saved_model.SaveOptions`, `optional`, default to `None`):
                :obj:`tf.saved_model.SaveOptions` object that specifies options for saving.
        
        .. note:: 
            
            Refers to `Signatures explanation <https://www.tensorflow.org/api_docs/python/tf/saved_model/save>`_
            from Tensorflow documentation for more information.

        Raises:
            ValueError: If `obj` is not trackable.
        """
        # noqa: E501 # pylint: enable=line-too-long
        try:
            import tensorflow as tf

            TF2 = tf.__version__.startswith("2")
        except ImportError:
            raise MissingDependencyException(
                "tensorflow is required by TensorflowModel"
            )

        if not _path_like(self._model):
            if TF2:
                tf.saved_model.save(
                    self._model, str(path), signatures=signatures, options=options,
                )
            else:
                if options:
                    logger.warning(
                        f"Parameter 'options: {str(options)}' is ignored when "
                        f"using tensorflow v{tf.__version__}"
                    )
                tf.saved_model.save(self._model, str(path), signatures=signatures)
        else:
            assert os.path.isdir(self._model)
            logger.warning(
                "Given model is detected as a path that contains SavedModel"
                "format. Running save will have no effects. Consider to use"
                "the given path to load from SavedModel format with TensorflowModel.load()"
            )
            return
