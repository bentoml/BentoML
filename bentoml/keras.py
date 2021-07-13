import importlib
import os

from bentoml.exceptions import (
    ArtifactLoadingException,
    InvalidArgument,
    MissingDependencyException,
)
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv
from bentoml.utils import cloudpickle

MODULE_NAME_FILE_ENCODING = "utf-8"
KERAS_MODEL_EXTENSION = ".h5"


class KerasModelArtifact(BentoServiceArtifact):
    """
    Artifact class for saving and loading Keras Model

    Args:
        name (string): name of the artifact
        default_custom_objects (dict): dictionary of Keras custom objects for model
        store_as_json_and_weights (bool): flag allowing storage of the Keras
            model as JSON and weights

    Raises:
        MissingDependencyException: keras or tensorflow.keras package is required for
            KerasModelArtifact
        InvalidArgument:  invalid argument type, model being packed must be instance of
            keras.engine.network.Network, tf.keras.models.Model, or their aliases

    Example usage:

    >>> from tensorflow import keras
    >>> from tensorflow.keras.models import Sequential
    >>> from tensorflow.keras.preprocessing import sequence, text
    >>>
    >>> model_to_save = Sequential()
    >>> # training model
    >>> model_to_save.compile(...)
    >>> model_to_save.fit(...)
    >>>
    >>> import bentoml
    >>> from bentoml.frameworks.keras import KerasModelArtifact
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts([KerasModelArtifact('model')])
    >>> class KerasModelService(bentoml.BentoService):
    >>>     @bentoml.api(input=JsonInput(), batch=False)
    >>>     def predict(self, parsed_json):
    >>>         input_data = text.text_to_word_sequence(parsed_json['text'])
    >>>         return self.artifacts.model.predict_classes(input_data)
    >>>
    >>> svc = KerasModelService()
    >>> svc.pack('model', model_to_save)
    """

    def __init__(
        self, name, default_custom_objects=None, store_as_json_and_weights=False,
    ):
        super(KerasModelArtifact, self).__init__(name)

        try:
            import tensorflow as tf
        except ImportError:
            raise MissingDependencyException(
                "Tensorflow package is required to use KerasModelArtifact. BentoML "
                "currently only support using Keras with Tensorflow backend."
            )

        self._store_as_json_and_weights = store_as_json_and_weights

        # By default assume using tf.keras module
        self._keras_module_name = tf.keras.__name__

        self._default_custom_objects = default_custom_objects
        self.graph = None
        self.sess = None

        self._model = None
        self._custom_objects = None

    def set_dependencies(self, env: BentoServiceEnv):
        # Note that keras module is not required, user can use tf.keras as an
        # replacement for the keras module. Although tensorflow module is required to
        #  be used as the default Keras backend
        if env._infer_pip_packages:
            pip_deps = ['tensorflow']
            if self._keras_module_name == 'keras':
                pip_deps.append('keras')
            env.add_pip_packages(pip_deps)

    def _keras_module_name_path(self, base_path):
        # The name of the keras module used, can be 'keras' or 'tensorflow.keras'
        return os.path.join(base_path, self.name + '_keras_module_name.txt')

    def _custom_objects_path(self, base_path):
        return os.path.join(base_path, self.name + '_custom_objects.pkl')

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + KERAS_MODEL_EXTENSION)

    def _model_weights_path(self, base_path):
        return os.path.join(base_path, self.name + '_weights.hdf5')

    def _model_json_path(self, base_path):
        return os.path.join(base_path, self.name + '_json.json')

    def bind_keras_backend_session(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise MissingDependencyException(
                "Tensorflow package is required to use KerasModelArtifact. BentoML "
                "currently only support using Keras with Tensorflow backend."
            )

        self.sess = tf.compat.v1.keras.backend.get_session()
        self.graph = self.sess.graph

    def create_session(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise MissingDependencyException(
                "Tensorflow package is required to use KerasModelArtifact. BentoML "
                "currently only support using Keras with Tensorflow backend."
            )

        self.graph = tf.compat.v1.get_default_graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        tf.compat.v1.keras.backend.set_session(self.sess)

    def pack(self, data, metadata=None):  # pylint:disable=arguments-renamed
        try:
            import tensorflow as tf
        except ImportError:
            raise MissingDependencyException(
                "Tensorflow package is required to use KerasModelArtifact. BentoML "
                "currently only support using Keras with Tensorflow backend."
            )

        if isinstance(data, dict):
            model = data['model']
            custom_objects = (
                data['custom_objects']
                if 'custom_objects' in data
                else self._default_custom_objects
            )
        else:
            model = data
            custom_objects = self._default_custom_objects

        if not isinstance(model, tf.keras.models.Model):
            error_msg = (
                "KerasModelArtifact#pack expects model argument to be type: "
                "keras.engine.network.Network, tf.keras.models.Model, or their "
                "aliases, instead got type: {}".format(type(model))
            )
            try:
                import keras

                if not isinstance(model, keras.engine.network.Network):
                    raise InvalidArgument(error_msg)
                else:
                    self._keras_module_name = keras.__name__
            except ImportError:
                raise InvalidArgument(error_msg)

        self.bind_keras_backend_session()
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
                        "Failed to import '{}' module when loading saved "
                        "KerasModelArtifact".format(keras_module_name)
                    )
        else:
            raise ArtifactLoadingException(
                "Failed to read keras model name from '{}' when loading saved "
                "KerasModelArtifact".format(self._keras_module_name_path(path))
            )

        if self._default_custom_objects is None and os.path.isfile(
            self._custom_objects_path(path)
        ):
            self._default_custom_objects = cloudpickle.load(
                open(self._custom_objects_path(path), 'rb')
            )

        if self._store_as_json_and_weights:
            # load keras model via json and weights if store_as_json_and_weights=True
            self.create_session()
            with self.graph.as_default():
                with self.sess.as_default():
                    with open(self._model_json_path(path), 'r') as json_file:
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
        return self.pack(model)

    def save(self, dst):
        # save the keras module name to be used when loading
        with open(self._keras_module_name_path(dst), "wb") as text_file:
            text_file.write(self._keras_module_name.encode(MODULE_NAME_FILE_ENCODING))

        # save custom_objects for model
        cloudpickle.dump(
            self._custom_objects, open(self._custom_objects_path(dst), "wb")
        )

        # save keras model using json and weights if requested
        if self._store_as_json_and_weights:
            with open(self._model_json_path(dst), "w") as json_file:
                json_file.write(self._model.to_json())
            self._model.save_weights(self._model_weights_path(dst))

        # otherwise, save standard keras model
        else:
            self._model.save(self._model_file_path(dst))

    def get(self):
        return self._model
