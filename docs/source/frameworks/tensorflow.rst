==========
Tensorflow
==========

Here's an example of serving Tensorflow models with BentoML:

.. code:: python

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
    tag = bentoml.tensorflow.save_model("native_tf_module", model)

    # retrieve metadata with `bentoml.models.get`:
    metadata = bentoml.models.get(tag)

    # `load` the model back in memory:
    model = bentoml.tensorflow.load_model("cancer_clf:latest")

    # Run a given model under `Runner` abstraction with `load_runner`
    _data = [[1.1, 2.2]]
    _tensor = tf.constant(_data)
    runner = bentoml.tensorflow.get("native_tf_module:latest").to_runner()
    runner.init_local()
    runner.run(_tensor)

.. note::

   You can find more examples for **Tensorflow** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.tensorflow

.. autofunction:: bentoml.tensorflow.save_model

.. autofunction:: bentoml.tensorflow.load_model

.. autofunction:: bentoml.tensorflow.get
