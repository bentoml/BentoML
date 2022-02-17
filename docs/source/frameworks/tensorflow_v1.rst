Tensorflow V1
-------------

Users can now use Tensorflow (v1 and v2 supported) with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

    import bentoml
    import tensorflow as tf
    import tempfile
    import numpy as np

    location = ""

    def simple_model_fn():
        x1 = tf.compat.v1.placeholder(shape=[None, 5], dtype=tf.float32, name="x1")
        x2 = tf.compat.v1.placeholder(shape=[None, 5], dtype=tf.float32, name="x2")
        factor = tf.compat.v1.placeholder(shape=(), dtype=tf.float32, name="factor")

        init = tf.constant_initializer([1.0, 1.0, 1.0, 1.0, 1.0])
        w = tf.Variable(init(shape=[5, 1], dtype=tf.float32))

        x = x1 + x2 * factor
        p = tf.matmul(x, w)
        return {"p": p, "x1": x1, "x2": x2, "factor": factor}

    simple_model = simple_model_fn()

    with tempfile.TemporaryDirectory() as temp_dir:
        with tf.compat.v1.Session() as sess:
            tf.compat.v1.enable_resource_variables()
            sess.run(tf.compat.v1.global_variables_initializer())
            inputs = {
                "x1": simple_model["x1"],
                "x2": simple_model["x2"],
                "factor": simple_model["factor"],
            }
            outputs = {"prediction": simple_model["p"]}

            tf.compat.v1.saved_model.simple_save(
                sess, temp_dir, inputs=inputs, outputs=outputs
            )
            location = temp_dir

    # `save` a given model and retrieve coresponding tag:
    tag = bentoml.tensorflow_v1.save("tf1_model", location)

    # retrieve metadata with `bentoml.models.get`:
    metadata = bentoml.models.get(tag)

    # `load` the model back in memory:
    model = bentoml.tensorflow_v1.load("tf1_model:latest")


    x = tf.convert_to_tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=tf.float32)
    f1 = tf.convert_to_tensor(3.0, dtype=tf.float32)
    f2 = tf.convert_to_tensor(2.0, dtype=tf.float32)

    # Run a given model under `Runner` abstraction with `load_runner`
    r1 = bentoml.tensorflow_v1.load_runner(
        tag,
        partial_kwargs=dict(factor=f1),
    )

    r2 = bentoml.tensorflow_v1.load_runner(
        tag,
        partial_kwargs=dict(factor=f2),
    )

    res = r1.run_batch(x1=x, x2=x)
    assert np.isclose(res[0][0], 60.0)
    res = r2.run_batch(x1=x, x2=x)
    assert np.isclose(res[0][0], 45.0)

We also offer :code:`import_from_tfhub` which enables users to import model from `Tensorflow Hub <https://tfhub.dev/>`_ and use it with BentoML:

.. code-block:: python

   import tensorflow_text as text
   import bentoml

   tag = bentoml.tensorflow_v1.import_from_tfhub("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

   # load model back with `load`:
   model = bentoml.tensorflow_v1.load(tag, load_as_wrapper=True)

.. note::

   You can find more examples for **Tensorflow** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.tensorflow_v1

.. autofunction:: bentoml.tensorflow_v1.save

.. autofunction:: bentoml.tensorflow_v1.load

.. autofunction:: bentoml.tensorflow_v1.load_runner

.. autofunction:: bentoml.tensorflow_v1.import_from_tfhub
