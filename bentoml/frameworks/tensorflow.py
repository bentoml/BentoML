import logging
import os
import pathlib
import shutil

from bentoml.exceptions import MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv

logger = logging.getLogger(__name__)


def _is_path_like(p):
    return isinstance(p, (str, bytes, pathlib.PurePath, os.PathLike))


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
            self._transform_input_by_tensorspec(arg, signatures[i])
            for i, arg in enumerate(args)
        )
        transformed_kwargs = {
            k: self._transform_input_by_tensorspec(arg, signatures_by_kw[k])
            for k, arg in kwargs.items()
        }
        if not self.concrete_func:
            self.concrete_func = self.origin_func.get_concrete_function()
        return self.concrete_func(*transformed_args, **transformed_kwargs)

    def __getattr__(self, k):
        return getattr(self.origin_func, k)

    @staticmethod
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

    @classmethod
    def hook_loaded_model(cls, loaded_model):
        try:
            from tensorflow.python.util import tf_inspect
            from tensorflow.python.eager import def_function
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

        self._wrapper = None

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_packages(['tensorflow'])

    def _saved_model_path(self, base_path):
        return os.path.join(base_path, self.name + '_saved_model')

    def pack(
        self, obj, signatures=None, options=None
    ):  # pylint:disable=arguments-differ
        """

        Args:
            obj: Either a path(str/byte/os.PathLike) containing exported
                `tf.saved_model` files, or a Trackable object mapping to the `obj`
                parameter of `tf.saved_model.save`
            signatures:
            options:
        """
        if _is_path_like(obj):
            self._wrapper = _ExportedTensorflowSavedModelArtifactWrapper(self, obj)
        else:
            self._wrapper = _TensorflowSavedModelArtifactWrapper(
                self, obj, signatures, options
            )

        return self

    def load(self, path):
        saved_model_path = self._saved_model_path(path)
        loaded_model = _load_tf_saved_model(saved_model_path)
        _TensorflowFunctionWrapper.hook_loaded_model(loaded_model)
        return self.pack(loaded_model)

    def save(self, dst):
        return self._wrapper.save(dst)

    def get(self):
        return self._wrapper.get()


class _ExportedTensorflowSavedModelArtifactWrapper:
    def __init__(self, tf_artifact, path):
        self.tf_artifact = tf_artifact
        self.path = path
        self.model = None

    def save(self, dst):
        # Copy exported SavedModel model directory to BentoML saved artifact directory
        shutil.copytree(self.path, self.tf_artifact._saved_model_path(dst))

    def get(self):
        if not self.model:
            self.model = _load_tf_saved_model(self.path)

        return self.model


class _TensorflowSavedModelArtifactWrapper:
    def __init__(self, tf_artifact, obj, signatures=None, options=None):
        self.tf_artifact = tf_artifact
        self.obj = obj
        self.signatures = signatures
        self.options = options

    def save(self, dst):
        try:
            import tensorflow as tf

            TF2 = tf.__version__.startswith('2')
        except ImportError:
            raise MissingDependencyException(
                "Tensorflow package is required to use TfSavedModelArtifact."
            )

        if TF2:
            return tf.saved_model.save(
                self.obj,
                self.tf_artifact._saved_model_path(dst),
                signatures=self.signatures,
                options=self.options,
            )
        else:
            if self.options:
                logger.warning(
                    "Parameter 'options: %s' is ignored when using Tensorflow "
                    "version 1",
                    str(self.options),
                )

            return tf.saved_model.save(
                self.obj,
                self.tf_artifact._saved_model_path(dst),
                signatures=self.signatures,
            )

    def get(self):
        return self.obj
