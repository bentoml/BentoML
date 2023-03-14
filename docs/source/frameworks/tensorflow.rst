==========
TensorFlow
==========

TensorFlow is an open source machine learning library focusing on deep neural networks. BentoML provides native support for 
serving and deploying models trained from TensorFlow.

Preface
-------

Even though ``bentoml.tensorflow`` supports Keras model, we recommend our users to use :doc:`bentoml.keras </frameworks/keras>` for better development experience. 

If you must use TensorFlow for your Keras model, make sure that your Keras model inference callback (such as ``predict``) is decorated with :obj:`~tf.function`.

.. note::

    - Keras is not optimized for production inferencing. There are `known reports <https://github.com/tensorflow/tensorflow/issues?q=is%3Aissue+sort%3Aupdated-desc+keras+memory+leak>`_ of memory leaks during serving at the time of BentoML 1.0 release. The same issue applies to ``bentoml.keras`` as it heavily relies on the Keras APIs.
    - Running Inference with :obj:`~bentoml.tensorflow` usually halves the time comparing with using ``bentoml.keras``.
    - ``bentoml.keras`` performs input casting that resembles the original Keras model input signatures.

.. note::

    :bdg-info:`Remarks:` We recommend users apply model optimization techniques such as **distillation** or **quantization**. Alternatively, Keras models can also be converted to :doc:`ONNX </frameworks/onnx>` models and leverage different runtimes.

Compatibility
-------------

BentoML requires TensorFlow version 2.0 or higher. For TensorFlow version 1.0, consider using a :ref:`concepts/runner:Custom Runner`.


Saving a Trained Model
----------------------

``bentoml.tensorflow`` supports saving ``tf.Module``, ``keras.models.Sequential``, and ``keras.Model``.

.. tab-set::

   .. tab-item:: tf.Module

      .. code-block:: python
        :caption: `train.py`

        # models created from the tf native API

        class NativeModel(tf.Module):
            def __init__(self):
                super().__init__()
                self.weights = np.asfarray([[1.0], [1.0], [1.0], [1.0], [1.0]])
                self.dense = lambda inputs: tf.matmul(inputs, self.weights)

            @tf.function(
                input_signature=[
                    tf.TensorSpec(shape=[None, 5], dtype=tf.float64, name="inputs")
                ]
            )
            def __call__(self, inputs):
                return self.dense(inputs)

        model = NativeModel()

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam()

        EPOCHS = 10
        for epoch in range(EPOCHS):
            with tf.GradientTape() as tape:
                predictions = model(train_x)
                loss = loss_object(train_y, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        bentoml.tensorflow.save(
            model,
            "my_tf_model",
            signatures={"__call__": {"batchable": True, "batch_dim": 0}}
        )

   .. tab-item:: keras.Model

      .. code-block:: python
        :caption: `train.py`

        class Model(keras.Model):
            def __init__(self):
                super().__init__()
                self.dense = keras.layers.Dense(1)

            @tf.function(
                input_signature=[
                    tf.TensorSpec(shape=[None, 5], dtype=tf.float64, name="inputs")
                ]
            )
            def call(self, inputs):
                return self.dense(inputs)

        model = Model()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_x, train_y, epochs=10)

        bentoml.tensorflow.save(
            model,
            "my_keras_model",
            signatures={"__call__": {"batchable": True, "batch_dim": 0}}
        )


   .. tab-item:: keras.model.Sequential

      .. code-block:: python
        :caption: `train.py`

        model = keras.models.Sequential(
            (
                keras.layers.Dense(
                    units=1,
                    input_shape=(5,),
                    dtype=tf.float64,
                    use_bias=False,
                    kernel_initializer=keras.initializers.Ones(),
                ),
            )
        )
        opt = keras.optimizers.Adam(0.002, 0.5)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(train_x, train_y, epochs=10)

        bentoml.tensorflow.save(
            model,
            "my_keras_model",
            signatures={"__call__": {"batchable": True, "batch_dim": 0}}
        )

   .. tab-item:: Functional keras.Model

      .. code-block:: python
        :caption: `train.py`

        x = keras.layers.Input((5,), dtype=tf.float64, name="x")
        y = keras.layers.Dense(
            6,
            name="out",
            kernel_initializer=keras.initializers.Ones(),
        )(x)
        model = keras.Model(inputs=x, outputs=y)
        opt = keras.optimizers.Adam(0.002, 0.5)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(train_x, train_y, epochs=10)

        bentoml.tensorflow.save(
            model,
            "my_keras_model",
            signatures={"__call__": {"batchable": True, "batch_dim": 0}}
        )

``bentoml.tensorflow`` also supports saving models that take multiple tensors as input:

.. code-block:: python
   :caption: `train.py`

   class MultiInputModel(tf.Module):
       def __init__(self):
           ...

       @tf.function(
           input_signature=[
               tf.TensorSpec(shape=[None, 5], dtype=tf.float64, name="x1"),
               tf.TensorSpec(shape=[None, 5], dtype=tf.float64, name="x2"),
               tf.TensorSpec(shape=(), dtype=tf.float64, name="factor"),
           ]
       )
       def __call__(self, x1: tf.Tensor, x2: tf.Tensor, factor: tf.Tensor):
           ...

   model = MultiInputModel()
   ... # training

   bentoml.tensorflow.save(
       model,
       "my_tf_model",
       signatures={"__call__": {"batchable": True, "batch_dim": 0}}
   )

.. note::

    :obj:`~bentoml.tensorflow.save_model` has two parameters: ``tf_signature`` and ``signatures``.

    Use the following arguments to define the model signatures to ensure consistent model behaviors in a Python session and from the BentoML model store:

    - ``tf_signatures`` is an alias to `tf.saved_model.save <https://www.tensorflow.org/api_docs/python/tf/saved_model/save>`_ *signatures* field. This optional signatures controls which methods in a given `obj <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/trackable/base.py#L281>`_ will be available to programs that consume `SavedModel's <https://www.tensorflow.org/guide/saved_model>`_, for example, serving APIs. Read more about TensorFlow's signatures behavior `from their API documentation <https://www.tensorflow.org/api_docs/python/tf/saved_model/save>`_.

    - ``signatures`` refers to a general :ref:`Model Signatures <concepts/model:Model Signatures>` that dictates which methods can be used for inference in the Runner context. This signatures dictionary will be used during the creation process of a Runner instance.

:bdg-info:`Note:` The signatures used for creating a Runner is ``{"__call__": {"batchable": False}}``.

This means BentoML’s :ref:`Adaptive Batching <guides/batching:Adaptive Batching>` is disabled when using :obj:`~bentoml.tensorflow.save_model()`.

If you want to utilize adaptive batching behavior and know your model's dynamic batching dimension, make sure to pass in ``signatures`` as follow: 


.. code-block:: python

    bentoml.tensorflow.save(model, "my_model", signatures={"__call__": {"batch_dim": 0, "batchable": True}})


Building a Service
------------------

Create a BentoML service with the previously saved `my_tf_model` pipeline using the :obj:`~bentoml.tensorflow` framework APIs.

.. code-block:: python
    :caption: `service.py`

    runner = bentoml.tensorflow.get("my_tf_model").to_runner()

    svc = bentoml.Service(name="test_service", runners=[runner])

    @svc.api(input=JSON(), output=JSON())
    async def predict(json_obj: JSONSerializable) -> JSONSerializable:
        batch_ret = await runner.async_run([json_obj])
        return batch_ret[0]

.. seealso::

    The following resources can help you to fine-tune your Tensorflow model:

    - |tf_function|_

    - Apply :ref:`Adaptive Batching <frameworks/tensorflow:Adaptive Batching>`.

    - `Serve on GPUs <https://www.tensorflow.org/guide/gpu>`_ and `GPU optimization <https://www.tensorflow.org/guide/gpu_performance_analysis>`_

    - `Graph optimization with Grappler <https://www.tensorflow.org/guide/graph_optimization>`_


.. _tf_function: https://www.tensorflow.org/guide/function

.. |tf_function| replace:: Performance tuning with well-defined ``tf.function``


Adaptive Batching
-----------------

Most TensorFlow models can accept batched data as input. If batch inference is supported, it is recommended to enable batching to take advantage of
the :ref:`adaptive batching <guides/batching:Adaptive Batching>` capability to improve the throughput and efficiency of the model.

Enable adaptive batching by overriding ``signatures`` argument with the method name and providing ``batchable`` and ``batch_dim`` configurations when saving the model to the model store:

.. code-block:: diff
   :caption: `batch.diff`

   diff --git a/train.py b/train_batched.py
   index 3b4bf11f..2d0ea09c 100644
   --- a/train.py
   +++ b/train_batched.py
   @@ -3,15 +3,24 @@ import bentoml
   class NativeModel(tf.Module):
       @tf.function(
           input_signature=[
   -            tf.TensorSpec(shape=[1, 5], dtype=tf.int64, name="inputs")
   +            tf.TensorSpec(shape=[None, 5], dtype=tf.float64, name="inputs")
           ]
       )
       def __call__(self, inputs):
           ...

   model = NativeModel()
   -bentoml.tensorflow.save(model, "test_model")
   +bentoml.tensorflow.save(
   +    model,
   +    "test_model",
   +    signatures={"__call__": {"batchable": True, "batch_dim": 0}},
   +)

   runner = bentoml.tensorflow.get("test_model")
   runner.init_local()
   +
   +#client 1
   runner.run([[1,2,3,4,5]])
   +
   +#client 2
   +runner.run([[6,7,8,9,0]])

From the diff above, when multiple clients send requests to a given server running this
model, BentoML will automatically batched inbound request and invoke ``model([[1,2,3,4,5], [6,7,8,9,0]])``

.. seealso::

   See :ref:`Adaptive Batching <guides/batching:Adaptive Batching>` to learn more about
   the adaptive batching feature in BentoML.

.. note::

   You can find more examples for **TensorFlow** in our :github:`bentoml/examples <bentoml/BentoML/tree/main/examples>` directory.

.. currentmodule:: bentoml.tensorflow
