==========
TensorFlow
==========

TensorFlow is an open source machine learning library focusing on deep neural networks. BentoML provides native support for 
serving and deploying models trained from TensorFlow.

Preface
-------

Even though ``bentoml.tensorflow`` supports Keras model, we recommend our users to use :ref:`bentoml.keras <frameworks/keras>` for better development experience. 

If you must use TensorFlow for your Keras model, make sure that your Keras model inference callback (such as ``predict``) is decorated with :code:`tf.function`.

.. note::

    - Keras is not optimized for production inferencing. There are `known reports <https://github.com/tensorflow/tensorflow/issues?q=is%3Aissue+sort%3Aupdated-desc+keras+memory+leak>`_ of memory leaks during serving at the time of BentoML 1.0 release. The same issue applies to ``bentoml.keras`` as it heavily relies on the Keras APIs.
    - Running Inference with :code:`bentoml.tensorflow` usually halves the time comparing with using ``bentoml.keras``.
    - ``bentoml.keras`` performs input casting that resembles the original Keras model input signatures.

.. note::

    :bdg-info:`Remarks:` We recommend users apply model optimization techniques such as **distillation** or **quantization**. Alternatively, Keras models can also be converted to :ref:`ONNX <frameworks/onnx>` models and leverage different runtimes (e.g. TensorRT, Apache TVM, etc.).

Compatibility
-------------

BentoML requires TensorFlow version 2.0 or higher. For TensorFlow version 1.0, consider using a :ref:`concepts/runner:Custom Runner`.


Saving a Trained Model
----------------------

``bentoml.tensorflow`` supports saving ``tf.Module``s, ``keras.models.Sequential``s, and ``keras.Model``s.

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

``bentoml.tensorflow`` also supports saving models that take multiple tensors as input:

.. code-block:: python
    :caption: `train.py`

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

.. note::

    :code:`bentoml.tensorflow.save_model` has two parameters: ``tf_signature`` and ``signatures``.
    Use the following arguments to define the model signatures signatures ensure consistent model behaviors in a Python session and from the BentoML model store.
    - `tf_signatures` is an alias to `tf.saved_model.save <https://www.tensorflow.org/api_docs/python/tf/saved_model/save>`_ *signatures* field. This optional signatures controls which methods in a given `obj <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/trackable/base.py#L281>`_ will be available to programs that consume `SavedModel's <https://www.tensorflow.org/guide/saved_model>`_, for example, serving APIs. Read more about TensorFlow's signatures behavior `from their API documentation <https://www.tensorflow.org/api_docs/python/tf/saved_model/save>`_.
    - ``signatures`` refers to a general :ref:`Model Signatures <concepts/model:Model Signatures>`_ that dictates which methods can be used for inference in the Runner context. This signatures dictionary will be used during the creation process of a Runner instance.

The signatures used for creating a Runner is ``{"__call__": {"batchable": False}}``. This means by default, BentoML’s `Adaptive Batching <guides/batching:Adaptive Batching>`_ is disabled when using :obj:`~bentoml.tensorflow.save_model()`. If you want to utilize adaptive batching behavior and know your model's dynamic batching dimension, make sure to pass in ``signatures`` as follow: 



.. code-block:: python

    bentoml.tensorflow.save(model, "my_model", signatures={"__call__": {"batch_dim": 0, "batchable": True}})


.. Step 2: Create & test a Runner
.. ------------------------------

.. .. code-block:: python

..     runner = bentoml.tensorflow.get("my_tf_model").to_runner()

..     runner.init_local()  # only for testing, do not call this in a bento service definition
..     runner.__call__.run(input_data)
..     # the same as:
..     # runner.run(input_data)


Building a Service
------------------

Create a BentoML service with the previously saved `my_tf_model` pipeline using the :code:`bentoml.tensorflow` framework APIs.

.. code-block:: python
    :caption: `service.py`

    runner = bentoml.tensorflow.get("my_tf_model").to_runner()

    svc = bentoml.Service(name="test_service", runners=[runner])

    @svc.api(input=JSON(), output=JSON())
    async def predict(json_obj: JSONSerializable) -> JSONSerializable:
        batch_ret = await runner.async_run([json_obj])
        return batch_ret[0]

.. note::

    Follow the steps to get the best performance out of your TensorFlow model.
    #. Save the model with well-defined :code:`tf.function` decorator.
    #. Apply adaptive batching if possible.
    #. Serve on GPUs if applicable.
    #. See performance guide from [TensorFlow Doc]

Adaptive Batching
-----------------

Most TensorFlow models can accept batched data as input. If batched interence is supported, it is recommended to enable batching to take advantage of 
the adaptive batching capability to improve the throughput and efficiency of the model. Enable adaptive batching by overriding the :code:`signatures` 
argument with the method name and providing :code:`batchable` and :code:`batch_dim` configurations when saving the model to the model store.

We may modify our code from

.. .. code-block:: python

..     class NativeModel(tf.Module):
..         @tf.function(
..             input_signature=[
..                 tf.TensorSpec(shape=[1, 5], dtype=tf.int64, name="inputs")
..             ]
..         )
..         def __call__(self, inputs):
..             ...

..     model = NativeModel()
..     bentoml.tensorflow.save(model, "test_model")  # the default signature is `{"__call__": {"batchable": False}}`


..     runner.run([[1,2,3,4,5]])  # -> bentoml will always call `model([[1,2,3,4,5]])`

.. to

.. code-block:: python
    :caption: `train.py`

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

.. seealso::

   See :ref:`Adaptive Batching <guides/batching:Adaptive Batching>` to learn more.


.. note::

   You can find more examples for **TensorFlow** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.tensorflow
