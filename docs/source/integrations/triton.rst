=======================
Triton Inference Server
=======================

*time expected: 10 minutes*

:github:`NVIDIA Triton Inference Server <triton-inference-server>` is a high performance, open-source inference server for serving deep learning models.
It is optimized to deploy models from multiple deep learning frameworks, including TensorRT,
TensorFlow, ONNX, to various deployments target and cloud providers. Triton is also designed with optimizations to maximize hardware utilization through concurrent model execution and efficient batching strategies.

BentoML now supports running Triton Inference Server as a :ref:`Runner <concepts/runner:Using Runners>`.
The following integration guide assumes that readers are familiar with BentoML architecture.
Check out our :ref:`tutorial <tutorial:Creating a Service>` should you wish to learn more about BentoML service definition.

For more information about Triton, please refer to the `Triton Inference Server documentation <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html>`_.

The code examples in this guide can also be found in the :github:`example folder <bentoml/BentoML/tree/main/examples/triton>`.

Why Integrating BentoML with Triton Inference Server?
~~~~~~~~~~~~~~~~~~~~~

If you are an existing Triton user, the integration provides simpler ways to add custom logics in Python, deploy distributed multi-model inference graph, unify model management across different ML frameworks and workflows, and standardise model packaging format with versioning and collaboration features. If you are an existing BentoML user, the integration improves the runner efficiency and throughput under high load thanks to Triton's efficient C++ runtime.

Prerequisites
~~~~~~~~~~~~~

Make sure to have at least BentoML 1.0.16:

.. code-block:: bash

    $ pip install -U "bentoml[triton]"

.. note::

   Triton Inference Server is currently only available in production mode (``--production`` flag) and will not work during development mode.

Additonally, you will need to have Triton Inference Server installed in your system. Refer to Triton's `building documentation <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/build.html>`_
to setup your environment. The recommended way to run Triton is through container (Docker/Podman). To pull the latest Triton container for testing, run:

.. code-block:: bash

    $ docker pull nvcr.io/nvidia/tritonserver:<yy>.<mm>-py3

.. note::

    ``<yy>.<mm>``: the version of Triton you wish to use. For example, at the time of writing, the latest version is ``23.01``.


Finally, The example Bento built from the example project with the :github:`YOLOv5 model <bentoml/BentoML/tree/main/examples/triton>` will be referenced throughout this guide.

.. note::

   To develop your own Bento with Triton, you can refer to the :github:`example folder <bentoml/BentoML/tree/main/examples/triton>` for more usage.

Get started with Triton Inference Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Triton Inference Server architecture evolves around the model repository and a inference server. The `model repository <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html>`_
is a filesystem based persistent volume that contains the models file and its respective `configuration <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html>`_ 
that defines how the model should be loaded and served. The inference server is implemented in either HTTP/REST or gRPC protocol to serve said models with various batching strategies.

BentoML provides a simple integration with Triton via :ref:`Runner <concepts/runner:Using Runners>`:

.. code-block:: python

   import bentoml

   triton_runner = bentoml.triton.Runner("triton_runner", model_repository="/path/to/model_repository")

The argument ``model_repository`` is the path to said model repository that Triton can use to serve the model. Note that ``model_repository`` also
supports S3 path:


.. code-block:: python

   import bentoml

   triton_runner = bentoml.triton.Runner("triton_runner",
                                         model_repository="s3://bucket/path/to/model_repository",
                                         cli_args=["--load-model=torchscrip_yolov5s", "--model-control-mode=explicit"]
   )

.. note::

   If models are saved on the file system, using the Triton runner requires setting up the model repository explicitly through the `includes` key in the `bentofile.yaml`.

.. note::

   The ``cli_args`` argument is a list of arguments that will be passed to the ``tritonserver`` command. For example, the ``--load-model`` argument is used to load a specific model from the model repository.
   See ``tritonserver --help`` for all available arguments.

From a developer perspective, remote invocation of Triton runners is similar to invoking any other BentoML runners. 

.. note::

   By default, ``bentoml.triton.Runner`` will run the ``tritonserver`` with gRPC protocol. To use HTTP/REST protocol, provide ``tritonserver_type=''http'`` to the ``Runner`` constructor.

   .. code-block:: python

      import bentoml

      triton_runner = bentoml.triton.Runner("triton_runner", model_repository="/path/to/model_repository", tritonserver_type="http")


Triton Runner Signatures
^^^^^^^^^^^^^^^^^^^^^^^^

Normally in a BentoML Runner, one can access the model signatures directly from the runners attributes. For example, the model signature ``predict``
of a ``iris_classifier_runner`` (see :ref:`service definition <tutorial:Creating a Service>`) can be accessed as ``iris_classifier_runner.predict.run``.

However, Triton runner's attributes represent individual models defined under the model repository. For example, if the model repository has the following structure:

.. code-block:: prolog

    model_repository
    ├── onnx_mnist
    │   ├── 1
    │   │   └── model.onnx
    │   └── config.pbtxt
    ├── tensorflow_mnist
    │   ├── 1
    │   │   └── model.savedmodel/
    │   └── config.pbtxt
    └── torchscript_mnist
        ├── 1
        │   └── model.pt
        └── config.pbtxt

Then each model inference can be accessed as ``triton_runner.onnx_mnist``, ``triton_runner.tensorflow_mnist``, or ``triton_runner.torchscript_mnist`` and invoked using either ``run`` or ``async_run``.

An example to demonstrate how to call the Triton runner:

.. code-block:: python

    import bentoml
    import numpy as np

    @svc.api(
        input=bentoml.io.Image.from_sample("./data/0.png"), output=bentoml.io.NumpyNdarray()
    )
    async def bentoml_torchscript_mnist_infer(im: Image) -> NDArray[t.Any]:
        arr = np.array(im) / 255.0
        arr = np.expand_dims(arr, (0, 1)).astype("float32")
        InferResult = await triton_runner.torchscript_mnist.async_run(arr)
        return InferResult.as_numpy("OUTPUT__0")

There are a few things to note here:

1. Triton runners should only be called within an API function. In other words, if ``triton_runner.torchscript_mnist.async_run`` is invoked in the
   global scope, it will not work. This is because Triton is not implemented natively in Python, and hence ``init_local`` is not supported.

   .. code-block:: python

       triton_runner.init_local()

       # TritonRunner 'triton_runner' will not be available for development mode.

2. ``async_run`` and ``run`` for any Triton runner call either takes all positional arguments or keyword arguments. The arguments
   should be in the same order as the `inputs/outputs <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#inputs-and-outputs>`_ signatures defined in ``config.pbtxt``.

   For example, if the following ``config.pbtxt`` is used for ``torchscript_mnist``:

   .. code-block:: protobuf

       platform: "pytorch_libtorch"
       dynamic_batching {}
       input {
        name: "INPUT__0"
        data_type: TYPE_FP32
        dims: -1
        dims: 1
        dims: 28
        dims: 28
       }
       input {
        name: "INPUT__1"
        data_type: TYPE_FP32
        dims: -1
        dims: 1
        dims: 28
        dims: 28
       }
       output {
        name: "OUTPUT__0"
        data_type: TYPE_FP32
        dims: -1
        dims: 10
       }
       output {
        name: "OUTPUT__1"
        data_type: TYPE_FP32
        dims: -1
        dims: 10
       }

   Then ``run`` or ``async_run`` takes either two positional arguments or two keyword arugments ``INPUT__0`` and ``INPUT__1``:

   .. code-block:: python

       # Both are valid
       triton_runner.torchscript_mnist.run(np.zeros((1, 28, 28)), np.zeros((1, 28, 28)))

       await triton_runner.torchscript_mnist.async_run(
           INPUT__0=np.zeros((1, 28, 28)), INPUT__1=np.zeros((1, 28, 28))
       )

   Mixing positional and keyword arguments will result in an error:

   .. code-block:: python

       triton_runner.torchscript_mnist.run(
           np.zeros((1, 28, 28)), INPUT__1=np.zeros((1, 28, 28))
       )
       # throws errors

3. ``run`` and ``async_run`` return a ``InferResult`` object. Regardless of the protocol used, the ``InferResult`` object has the following methods:

   - ``as_numpy(name: str) -> NDArray[T]``: returns the result as a numpy array. The argument is the name of the output defined in ``config.pbtxt``.

   - ``get_output(name: str) -> InferOutputTensor | dict[str, T]``: Returns the results as a |infer_output_tensor|_ (gRPC) or 
     a dictionary (HTTP). The argument is the name of the output defined in ``config.pbtxt``.

   - ``get_response(self) -> ModelInferResponse | dict[str, T]``: Returns the entire response as a |model_infer_response|_ (gRPC) or 
     a dictionary (HTTP).

   Using the above ``config.pbtxt`` as example, the model consists of two outputs, ``OUTPUT__0`` and ``OUTPUT__1``.

   To get ``OUTPUT__0`` as a numpy array:

   .. tab-set::

      .. tab-item:: gRPC
         :sync: grpc

         .. code-block:: python

             InferResult = triton_runner.torchscript_mnist.run(np.zeros((1, 28, 28)), np.zeros((1, 28, 28)))
             return InferResult.as_numpy("OUTPUT__0")

      .. tab-item:: HTTP
         :sync: http

         .. code-block:: python

             InferResult = triton_runner.torchscript_mnist.run(np.zeros((1, 28, 28)), np.zeros((1, 28, 28)))
             return InferResult.as_numpy("OUTPUT__0")

   To get ``OUTPUT__1`` as a JSON dictionary:

   .. tab-set::

      .. tab-item:: gRPC
         :sync: grpc

         .. code-block:: python

             InferResult = triton_runner.torchscript_mnist.run(np.zeros((1, 28, 28)), np.zeros((1, 28, 28)))
             return InferResult.get_output("OUTPUT__0", as_json=True)

      .. tab-item:: HTTP
         :sync: http

         .. code-block:: python

             InferResult = triton_runner.torchscript_mnist.run(np.zeros((1, 28, 28)), np.zeros((1, 28, 28)))
             return InferResult.get_output("OUTPUT__0")

Additonally, the Triton runner exposes all `tritonclient <https://github.com/triton-inference-server/client>`_ functions.

.. dropdown:: Supported client APIs
    :icon: triangle-down

    The list below comprises all the model management APIs from ``tritonclient`` that are supported by Triton runners:

    - ``get_model_config``
    - ``get_model_metadata``
    - ``get_model_repository_index``
    - ``is_model_ready``
    - ``is_server_live``
    - ``is_server_ready``
    - ``load_model``
    - ``unload_model``
    - ``infer``
    - ``stream_infer``

    The following advanced client APIs are also supported:

    - ``get_cuda_shared_memory_status``
    - ``get_inference_statistics``
    - ``get_log_settings``
    - ``get_server_metadata``
    - ``get_system_shared_memory_status``
    - ``get_trace_settings``
    - ``register_cuda_shared_memory``
    - ``register_system_shared_memory``
    - ``unregister_cuda_shared_memory``
    - ``unregister_system_shared_memory``
    - ``update_log_settings``
    - ``update_trace_settings``

.. epigraph::
   :bdg-primary:`Important:` All of the client APIs are asynchronous. To use them, make sure to use it under an async ``@svc.api``. See :ref:`concepts/service:Sync vs Async APIs`

   .. code-block:: python
      :caption: `service.py`

      @svc.api(input=bentoml.io.Text.from_sample("onnx_mnist"), output=bentoml.io.JSON())
      async def unload_model(input_model: str):
          await triton_runner.unload_model(input_model)
          return {"unloaded": input_model}

Packaging BentoService with Triton Inference Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build your BentoService with Triton Inference Server, add the following to your ``bentofile.yaml``
or use :ref:`reference/core:bentoml.bentos.build`:

.. tab-set::

   .. tab-item:: Building with ``bentofile.yaml``

      .. literalinclude:: ./snippets/triton/bentofile.yaml
         :language: yaml
         :caption: `bentofile.yaml`

      Building this Bento with :ref:`bentoml build <reference/cli:build>`:

      .. code-block:: bash

         $ bentoml build

   .. tab-item:: Building with ``bentoml.bentos.build``

      .. literalinclude:: ./snippets/triton/build_bento.py
         :language: python
         :caption: `build_bento.py`

Notice that we are using ``nvcr.io/nvidia/tritonserver:22.12-py3`` as our base image. This can be substituted with any other
custom base image that has ``tritonserver`` binary available. See Triton's documentation `here <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/build.html>`_ 
to learn more about building/composing custom Triton image.

.. epigraph::
    :bdg-primary:`Important:` The provided Triton image from NVIDIA includes Python 3.8. Therefore, if you are developing your Bento
    with any other Python version, make sure that your ``service.py`` is compatible with Python 3.8.

.. tip::

   To see all available options for Triton run:

   .. code-block:: bash

      $ docker run --init --rm -p 3000:3000 triton-integration:gpu tritonserver --help

Current Caveats
~~~~~~~~~~~~~~~

At the time of writing, there are a few caveats that you should be aware of when using TritonRunner:

Versioning Policy Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, model configuration `version policy <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#version-policy>`_
is set to ``latest(n=1)``, meaning the latest version of the model will be loaded into Triton server.

Currently, TritonRunner only supports the ``latest`` policy.
If you have multiple versions of the same model in your BentoService, then the runner only consider the latest version.

For example, if the model repository have the following structure:

.. code-block:: bash

    model_repository
    ├── onnx_mnist
    │   ├── 1
    │   │   └── model.onnx
    │   ├── 2
    │   │   └── model.onnx
    │   └── config.pbtxt
    ...

.. epigraph::
   Then ``triton_runner.onnx_mnist`` will reference to the latest version of the model (in this case, version 2).

To use a specific version of said model, refer to the example below:

.. literalinclude:: ./snippets/triton/service_model_version_1.py
    :language: python
    :caption: `service.py`

Inference Protocol and Metrics Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, TritonRunner uses `the Inference protocol <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/inference_protocols.html?highlight=inference%20api>`_ for both REST and gRPC.

HTTP/REST APIs is disabled by default, though it can be enabled when creating the runner by passing ``tritonserver_type`` to the Runner:

.. code-block:: python

    triton_runner = TritonRunner(
        "http_runner",
        "/path/to/model_repository",
        tritonserver_type="http"
    )

.. epigraph::
    Currently, TritonRunner does not support running `Metrics server <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html>`_.
    If you are interested in supporting the metrics server, please open an issue on :github:`GitHub <bentoml/BentoML/issues/new/choose>`

Additionally, BentoML will allocate a random port for the gRPC/HTTP server, hence ``grpc-port`` or ``http-port`` options that is passed to Runner ``cli_args`` will be omitted.

Adaptive Batching
^^^^^^^^^^^^^^^^^

:ref:`Adaptive batching <guides/batching:Adaptive Batching>` is a feature supported by BentoML runners that allows for efficient batch size selection during inference. However, it's important to note that this feature is not compatible with ``TritonRunner``.

``TritonRunner`` is designed as a standalone Triton server, which means that the adaptive batching logic in BentoML runners is not invoked when using ``TritonRunner``.

Fortunately, Triton supports its own solution for efficient batching called `dynamic batching <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#scheduling-and-batching>`_.
Similar to adaptive batching, dynamic batching also allows for the selection of the optimal batch size during inference. To use dynamic batching in Triton, relevant settings can be specified in the
`model configuration <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#model-configuration>`_ file.

.. admonition:: 🚧 Help us improve the integration!

    This integration is still in its early stages and we are looking for feedbacks and contributions to make it even better!

    If you have any feedback or want to contribute any improvements to the
    Triton Inference Server integration, we would love to see your `feature requests <https://github.com/bentoml/BentoML/issues>`_
    and `pull request! <https://github.com/bentoml/BentoML/pulls>`_

    Check out the `BentoML development guide <https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md>`_
    and `documentation guide <https://github.com/bentoml/BentoML/blob/main/docs/README.md>`_
    to get started.

.. _infer_output_tensor: https://github.com/triton-inference-server/common/blob/7b37a244e939c69c51df74d31cd905ae6ffec2f9/protobuf/grpc_service.proto#L703

.. |infer_output_tensor| replace:: :code:`InferOutputTensor`

.. _model_infer_response: https://github.com/triton-inference-server/common/blob/7b37a244e939c69c51df74d31cd905ae6ffec2f9/protobuf/grpc_service.proto#L696

.. |model_infer_response| replace:: :code:`ModelInferResponse`
