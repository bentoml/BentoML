Tensorflow
----------

Users can now use Tensorflow (v1 and v2 supported) with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

    import bentoml
    import tensorflow as tf

    class NativeModel(tf.Module):
        def __init__(self):
        super().__init__()
        self.weights = np.asfarray([[1.0], [1.0], [1.0], [1.0], [1.0]])
        self.dense = lambda inputs: tf.matmul(inputs, self.weights)

        @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, 5], dtype=tf.float64, name="inputs")]
        )
        def __call__(self, inputs):
        return self.dense(inputs)

    model = NativeModel()

    # `save` a given model and retrieve coresponding tag:
    tag = bentoml.tensorflow.save("native_tf_module", model)

    # retrieve metadata with `bentoml.models.get`:
    metadata = bentoml.models.get(tag)

    # `load` the model back in memory:
    model = bentoml.tensorflow.load("cancer_clf:latest")

    # Run a given model under `Runner` abstraction with `load_runner`
    _data = [[1.1, 2.2]]
    _tensor = tf.constant(_data)
    runner = bentoml.tensorflow.load_runner("native_tf_module:latest"")
    runner.run(_tensor)

We also offer :code:`import_from_tfhub` which enables users to import model from `Tensorflow Hub <https://tfhub.dev/>`_ and use it with BentoML:

.. code-block:: python

   import tensorflow_text as text
   import bentoml

   tag = bentoml.tensorflow.import_from_tfhub("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

   # load model back with `load`:
   model = bentoml.tensorflow.load(tag, load_as_wrapper=True)

.. note::

   You can find more examples for **Tensorflow** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.tensorflow

.. autofunction:: bentoml.tensorflow.save

.. autofunction:: bentoml.tensorflow.load

.. autofunction:: bentoml.tensorflow.load_runner

.. autofunction:: bentoml.tensorflow.import_from_tfhub
