import logging
import os
import pathlib
import shutil
import tempfile

from bentoml.exceptions import MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv

logger = logging.getLogger(__name__)


def _is_path_like(p):
    return isinstance(p, (str, bytes, pathlib.PurePath, os.PathLike))


def _transform_input_by_tensorspec(_input, tensorspec):
    '''
    transform dtype & shape following tensorspec
    '''
    try:
        import tensorflow as tf
    except ImportError:
        raise MissingDependencyException(
            "Tensorflow package is required to use TfSavedModelArtifact"
        )

    if _input.dtype != tensorspec.dtype:
        # may raise TypeError
        _input = tf.dtypes.cast(_input, tensorspec.dtype)
    return _input


class _TensorflowFunctionWrapper:
    '''
    TensorflowFunctionWrapper
    transform input tensor following function input signature
    '''

    def __init__(self, origin_func, fullargspec):
        self.origin_func = origin_func
        self.concrete_func = None
        self.fullargspec = fullargspec
        self._args_to_indices = {arg: i for i, arg in enumerate(fullargspec.args)}

    def __call__(self, *args, **kwargs):
        signatures = self.origin_func.input_signature
        if signatures is None:
            return self.origin_func(*args, **kwargs)

        for k in kwargs:
            if k not in self._args_to_indices:
                raise TypeError(f"Function got an unexpected keyword argument {k}")
        signatures_by_kw = {k: signatures[self._args_to_indices[k]] for k in kwargs}
        # INFO:
        # how signature with kwargs works?
        # https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/eager/function.py#L1519

        transformed_args = tuple(
            _transform_input_by_tensorspec(arg, signatures[i])
            for i, arg in enumerate(args)
        )
        transformed_kwargs = {
            k: _transform_input_by_tensorspec(arg, signatures_by_kw[k])
            for k, arg in kwargs.items()
        }
        if not self.concrete_func:
            self.concrete_func = self.origin_func.get_concrete_function()
        return self.concrete_func(*transformed_args, **transformed_kwargs)

    def __getattr__(self, k):
        return getattr(self.origin_func, k)

    @classmethod
    def hook_loaded_model(cls, loaded_model):
        try:
            from tensorflow.python.eager import def_function
            from tensorflow.python.util import tf_inspect
        except ImportError:
            raise MissingDependencyException(
                "Tensorflow package is required to use TfSavedModelArtifact"
            )

        for k in dir(loaded_model):
            v = getattr(loaded_model, k, None)
            if isinstance(v, def_function.Function):
                fullargspec = tf_inspect.getfullargspec(v)
                setattr(loaded_model, k, cls(v, fullargspec))


def _load_tf_saved_model(path):
    try:
        import tensorflow as tf
        from tensorflow.python.training.tracking.tracking import AutoTrackable

        TF2 = tf.__version__.startswith('2')
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
                '''Importing SavedModels from TensorFlow 1.x.
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
                '''
            )
        return loaded


class TensorflowSavedModelArtifact(BentoServiceArtifact):
    """
    Abstraction for saving/loading Tensorflow model in tf.saved_model format

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
        super(TensorflowSavedModelArtifact, self).__init__(name)

        self.model = None
        self._tmpdir = None
        self._path = None

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_packages(['tensorflow'])

    def _saved_model_path(self, base_path):
        return os.path.join(base_path, self.name + '_saved_model')

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

                TF2 = tf.__version__.startswith('2')
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

        return self

    def get(self):
        if self.model is None:
            loaded_model = _load_tf_saved_model(self._path)
            _TensorflowFunctionWrapper.hook_loaded_model(loaded_model)
            self.model = loaded_model

        return self.model

    def load(self, path):
        saved_model_path = self._saved_model_path(path)
        return self.pack(saved_model_path)

    def save(self, dst):
        # Copy exported SavedModel model directory to BentoML saved artifact directory
        shutil.copytree(self._path, self._saved_model_path(dst))

    def __del__(self):
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
