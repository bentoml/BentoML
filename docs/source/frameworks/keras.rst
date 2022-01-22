Keras
-----

| Deep learning for humans.
| Keras is an API designed for human beings, not machines.
| Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs,
| it minimizes the number of user actions required for common use cases, and it provides clear &
| actionable error messages. It also has extensive documentation and developer guides. - `Source <https://keras.io/>`_

Users can now use Keras (with Tensorflow v1 and v2 backend) with BentoML with the following three API: :code:`load`, :code:`save`, and :code:`load_runner` as follows:

.. tabs::

   .. code-tab:: keras_v1

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
            )
         )

         opt = keras.optimizers.Adam(0.002, 0.5)
         net.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
         return net
      
      model = KerasSequentialModel()

      # `save` a given model and retrieve coresponding tag:
      tag = bentoml.keras.save("keras_model", model, store_as_json=True)

      # retrieve metadata with `bentoml.models.get`:
      metadata = bentoml.models.get(tag)

      # retrieve session that save keras model with `bentoml.keras.get_session()`:
      session = bentoml.keras.get_session()
      session.run(tf.global_variables_initializer())
      with session.as_default():
         # `load` the model back in memory:
         loaded = bentoml.keras.load("keras_model:latest")

   .. code-tab:: keras_v2

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
            )
         )

         opt = keras.optimizers.Adam(0.002, 0.5)
         net.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
         return net
      
      model = KerasSequentialModel()

      # `save` a given model and retrieve coresponding tag:
      tag = bentoml.keras.save("keras_model", model, store_as_json=True)

      # retrieve metadata with `bentoml.models.get`:
      metadata = bentoml.models.get(tag)

      # `load` the model back in memory:
      loaded = bentoml.keras.load("keras_model:latest")

.. note::
   You can find more examples for **Keras** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.keras

.. autofunction:: bentoml.keras.save

.. autofunction:: bentoml.keras.load

.. autofunction:: bentoml.keras.load_runner

.. autofunction:: bentoml.keras.get_session