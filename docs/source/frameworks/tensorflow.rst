==========
Tensorflow
==========

Before everything
-----------------

- Are you trying to save a keras model?
  You may wanna use :ref:`frameworks/keras:Keras` instead, or make sure the keras model is well signated with `tf.function` `see the difference`_
- Are you using tensorflow v1?
  use `bentoml.tensorflow_v1`_ instead
- Want more better performance?
  - do some **Model Distilling** or **Model Quantilization**
  - convert to ONNX models, save them with `bentoml.onnx`_, use proper backend (for eg: TensorRT)


.. seealso::
    Difference between using bentoml.keras and bentoml.tensorflow
  * keras is not optimized for inference in production. There’re some reports about memory leaking during serving. bentoml.keras should have same concerns since it rely on the the keras API + Tensorflow backend.
  * inference with bentoml.tensorflow is about 2x faster than bentoml.keras while being correctly decorated with tf.function
  * bentoml.keras would do input casting just like the original keras model object. Which means using it is more convinient in debuging. No need to `


Compatibility
-------------

bentoml.tensorflow requires Tensorflow v2.0 or higher.


Step 1: Saving a trained model
------------------------------

`bentoml.tensorflow` supports saving a model in the following format:

.. tab-set::

   .. tab-item:: tf.Module

      .. code-block:: python

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

   .. tab-item:: keras Sequential

      .. code-block:: python

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

   .. tab-item:: keras Functional

      .. code-block:: python
    
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
