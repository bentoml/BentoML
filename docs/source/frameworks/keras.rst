=====
Keras
=====

Users can now use Keras (with Tensorflow v2 backend) with BentoML with the following
APIs:

.. code:: python

    import bentoml
    import tensorflow as tf
    import tensorflow.keras as keras

    def KerasSequentialModel() -> keras.models.Model:
        net = keras.models.Sequential(
            (
                keras.layers.Dense(
                    units=1,
                    input_shape=(5,),
                    use_bias=False,
                    kernel_initializer=keras.initializers.Ones(),
                ),
            ),
        )

        opt = keras.optimizers.Adam(0.002, 0.5)
        net.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        return net

    model = KerasSequentialModel()

    # save a given model:
    tag = bentoml.keras.save_model("keras_model", model)

    # retrieve metadata with the tag:
    metadata = bentoml.models.get(tag)

    # load the model back in memory:
    loaded = bentoml.keras.load_model("keras_model:latest")

.. note::

   You can find more examples for **Keras** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.keras

.. autofunction:: bentoml.keras.save_model

.. autofunction:: bentoml.keras.load_model

.. autofunction:: bentoml.keras.get
