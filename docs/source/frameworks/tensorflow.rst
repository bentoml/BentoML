==========
TensorFlow
==========

TensorFlow is an open source machine learning library focusing on training on inference of deep neural networks. BentoML provides native support for 
serving and deploying models trained from TensorFlow.

Preface
-------

Consider using BentoML :ref:`frameworks/keras:Keras` framework if working with a Keras model. If continuing with TensorFlow, make sure the keras model is well signated with `tf.function` `see the difference`_

.. note::

    - Keras is not optimized for inference in production. There are reports about memory leaks during serving. :code:`bentoml.keras` has same concerns since it rely on the the Keras APIs.
    - Inference with :code:`bentoml.tensorflow` is about twice faster than :code:`bentoml.keras`.
    - :code:`bentoml.keras` performs input casting similar to the original Keras model for better debugging experiences.

To improve the performance of Keras models, consider applying techniques like **model distillation** or **model quantization**. Alternatively, the Keras model can be converted to a ONNX model and saved with :code:`bentoml.onnx` to leverage better performance runtimes (for eg: TensorRT).

Compatibility
-------------

BentoML requires TensorFlow version 2.0 or higher. For TensorFlow version 1.0, consider using a :ref:`concepts/runner:Custom Runner`.


Saving a trained model
----------------------

`bentoml.tensorflow` supports saving a model in the formats of :code:`tf.Module`, :code:`keras.models.Sequential`, or :code:`keras.Model`.

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
        ... # training

        # =========

        bentoml.tensorflow.save(model, "my_tf_model")

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

        bentoml.tensorflow.save(model, "my_keras_model")

   .. tab-item:: keras.Model (Functional?)

      .. code-block:: python
        :caption: `train.py`
    
        x = keras.layers.Input((5,), dtype=tf.float64, name="x")
        y = keras.layers.Dense(
            6,
            name="out",
            kernel_initializer=keras.initializers.Ones(),
        )(x)
        model = keras.Model(inputs=x, outputs=y)

        bentoml.tensorflow.save(model, "my_keras_model")


Sometime a model may take multiple tensors as input

.. code-block:: python

   class MultiInputModel(tf.Module):
        def __init__(self):
            ...

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[1, 5], dtype=tf.float64, name="x1"),
                tf.TensorSpec(shape=[1, 5], dtype=tf.float64, name="x2"),
                tf.TensorSpec(shape=(), dtype=tf.float64, name="factor"),
            ]
        )
        def __call__(self, x1: tf.Tensor, x2: tf.Tensor, factor: tf.Tensor):
            ...

    model = MultiInputModel()
    ... # training

    bentoml.tensorflow.save(model, "my_tf_model")

.. seealso::
   `bentoml.tensorflow.save`_ has two parameters `tf_signature` and `signatures`.
   They are important when you want to save a model with a ensured behavior.
   The `tf_signature` is a dict of tensor names and their shapes, which adaopted from
   the signature of tensorflow.saved_model.save. You may find more details about it
   on the `tensorflow.saved_model.save`_ documentation.
   The `signatures` is a dict of functions names and some other information.
   If you know your model has a dynamic batch dimension, you can use `signatures` to tell
   bentoml about that for possible future optimizations like this:

   bentoml.tensorflow.save(model, "my_model", signatures={"__call__": {"batch_dim": 0, "batchable": True}})


Step 2: Create & test a Runner
------------------------------

.. code-block:: python

    runner = bentoml.tensorflow.get("my_tf_model").to_runner()

    runner.init_local()  # only for testing, do not call this in a bento service definition
    runner.__call__.run(input_data)
    # the same as:
    # runner.run(input_data)


Step 3: Building a Service using the runner
-------------------------------------------

.. code-block:: python

    runner = bentoml.tensorflow.get("my_tf_model").to_runner()

    svc = bentoml.Service(name="test_service", runners=[runner])

    @svc.api(input=JSON(), output=JSON())
    async def predict(json_obj: JSONSerializable) -> JSONSerializable:
        batch_ret = await runner.async_run([json_obj])
        return batch_ret[0]


Read More: Performance Guide
----------------------------

To boost your service

1. save the model with well defined tf.function decorator
2. apply # micro-batching if possible
3. use Nvidia GPUs if possible
4. other performance guide from [Tensorflow Doc]



Read More: Adaptive batching
----------------------------

If the model can take batch data as the input(quiet common), we can enable the micro-batching feature for higher throuput.

We may modify our code from

.. code-block:: python

    class NativeModel(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[1, 5], dtype=tf.int64, name="inputs")
            ]
        )
        def __call__(self, inputs):
            ...

    model = NativeModel()
    bentoml.tensorflow.save(model, "test_model")  # the default signature is `{"__call__": {"batchable": False}}`


    runner.run([[1,2,3,4,5]])  # -> bentoml will always call `model([[1,2,3,4,5]])`

to

.. code-block:: python

    class NativeModel(tf.Module):

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, 5], dtype=tf.float64, name="inputs")
            ]
        )
        def __call__(self, inputs):
            ...

    model = NativeModel()
    bentoml.tensorflow.save(
        model,
        "test_model",
        signatures={"__call__": {"batchable": True, "batchdim": 0}},
    )

    #client 1
    runner.run([[1,2,3,4,5]])

    #client 2
    runner.run([[6,7,8,9,0]])

    # if multiple requests from different clients arrived at the same time,
    # bentoml will automatically merge them and call model([[1,2,3,4,5], [6,7,8,9,0]])




.. note::

   You can find more examples for **Tensorflow** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.tensorflow
