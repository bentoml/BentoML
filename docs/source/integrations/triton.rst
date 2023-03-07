=======================
Triton Inference Server
=======================

*time expected: 12 minutes*

NVIDIA Triton Inference Server [#triton]_ is a high performance, open-source inference server for serving deep learning models.
It is optimized to deploy models from multiple deep learning frameworks, including TensorRT,
TensorFlow, ONNX, to various deployments target and cloud providers. Triton is also designed with 
optimizations to maximize hardware utilization through concurrent model execution and efficient batching strategies.

BentoML now supports running Triton Inference Server through the :ref:`Runner <concepts/runner:Using Runners>`
architecture. The following integration guide makes the assumption that readers are familiar with BentoML infrastructure.
Make sure to check out our :ref:`tutorial <tutorial:Creating a Service>` should you wish to learn more about BentoML.

The guide will try to be as comprehensive and detailed as possible, yet all the features from Triton Inference Server will not be covered. For more information, please refer to their documentation [#triton_docs]_.

The code examples in this guide can also be found in the examples folder [#triton_runner]_.


Prerequisites
~~~~~~~~~~~~~

Make sure to have at least BentoML 1.0.14 and ``tritonclient`` at least version 2.29.0 available in your Python environment:

.. code-block:: bash

    $ pip install -U bentoml tritonclient[all]

.. note::

   Triton Inference Server is currently only available in production mode (``--production`` flag) and will not work during development mode.

Additonally, you will need to have Triton Inference Server installed in your system. Refer to Triton's building documentation [#triton_build]_
to setup your environment.

The recommended way to run Triton is through container (Docker/Podman). To pull the latest Triton container for testing, run:

.. code-block:: bash

    $ docker pull nvcr.io/nvidia/tritonserver:<yy>.<mm>-py3

.. note::

    ``<yy>.<mm>``: the version of Triton you wish to use. For example, at the time of writing, the latest version is ``22.12``.

In this guide, we will demonstrate the capabilities of Triton Inference Server with BentoML and how one can take advantages of both frameworks.

Finally, The example Bento built from the example project with YoloV5 model [#triton_runner]_ will be referenced throughout this guide.

.. note::

   To develop your own Bento with Triton, you can refer to the example folder [#triton_runner]_ for more usage.

Why do you want this?
~~~~~~~~~~~~~~~~~~~~~

* For existing Triton users who are searching for a simple way to add pre/post-processing logics, comprehensive supports of multi-model inference graph
  and a standardisation for your model packaging format, which can then be easily reused and collaborated with other teams members.

* For existing Triton users who are looking to unify model management with other machine learning frameworks/workflow.

* For existing BentoML users, Triton reduces the performance gap between model server written in C++ and Python. While BentoML
  is constantly improving, Triton provides better performance under given certain conditions.

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

   triton_runner = bentoml.triton.Runner("triton_runner", model_repository="s3://bucket/path/to/model_repository")

.. note::

   If models are saved on the file system, using the Triton runner requires setting up the model repository explicitly through the `includes` key in the `bentofile.yaml`.

From a developer perspective, remote invocation of Triton runners is similar to invoking any other BentoML runners. 

.. note::

   By default, ``bentoml.triton.Runner`` will run the ``tritonserver`` with gRPC protocol. To use HTTP/REST protocol, provide ``tritonserver_type=''http'`` to the ``Runner`` constructor.

   .. code-block:: python

      import bentoml

      triton_runner = bentoml.triton.Runner("triton_runner", model_repository="/path/to/model_repository", tritonserver_type="http")


Triton Runner signatures
^^^^^^^^^^^^^^^^^^^^^^^^

Normally in a BentoML Runner, one can access the model signatures directly from the runners attributes. For example, the model signature ``predict``
of a ``iris_classifier_runner`` (as often seen in our :ref:`tutorial <tutorial:Creating a Service>`) can be accessed as ``iris_classifier_runner.predict.run``.

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

1. Triton runners should only be called **lazily**. In other words, if ``triton_runner.torchscript_mnist.async_run`` is invoked in the
   global scope, it will not work. This is because Triton is not implemented natively in Python, and hence ``init_local`` is not supported.

   .. code-block:: python

       triton_runner.init_local()

       # TritonRunner 'triton_runner' will not be available for development mode.

2. ``async_run`` and ``run`` for any Triton runner call either takes all positional arguments or keyword arguments. The arguments
   should be in the same order as the inputs/outputs [#triton_inputs_outputs]_ signatures defined in ``config.pbtxt``.

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

   The following will result in an error:

   .. code-block:: python

       triton_runner.torchscript_mnist.run(
           np.zeros((1, 28, 28)), INPUT__1=np.zeros((1, 28, 28))
       )
       # throws errors

3. ``run`` and ``async_run`` return a ``InferResult`` object. Regardless of the protocol used, the ``InferResult`` object has the following methods:

   - ``as_numpy(name: str) -> NDArray[T]``: returns the result as a numpy array. The argument is the name of the output defined in ``config.pbtxt``.

   - ``get_output(name: str) -> InferOutputTensor | dict[str, T]``: Returns the results as a ``InferOutputTensor`` [#infer_output_tensor]_ (gRPC) or 
     a dictionary (HTTP). The argument is the name of the output defined in ``config.pbtxt``.

   - ``get_response(self) -> ModelInferResponse | dict[str, T]``: Returns the entire response as a ``ModelInferResponse`` [#model_infer_response]_ (gRPC) or 
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

   To get ``OUTPUT__1`` as a json dictionary:

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

Additonally, Triton runners exposes all `tritonclient <https://github.com/triton-inference-server/client>`_ to TritonRunner.

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
custom base image that has ``tritonserver`` binary available. See Triton's documentation [#triton_build]_ to learn more about building/composing custom Triton image.

.. epigraph::
    :bdg-primary:`Important:` The provided Triton image from NVIDIA includes Python 3.8. Therefore, if you are developing your Bento
    with any other Python version, make sure that your ``service.py`` is compatible with Python 3.8.

Serving BentoService with Triton Inference Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After containerizing with :ref:`bentoml containerize <reference/cli:containerize>`, :ref:`serve <reference/cli:serve>`
command now takes in additional ``--triton-options`` argument to pass options for ``tritonserver``:

.. tab-set::

    .. tab-item:: MacOS/Windows
       :sync: macwin

       .. tab-set::

          .. tab-item:: GPU
              :sync: gpu

              .. code-block:: bash

                  $ docker run --init --rm -p 3000:3000 triton-integration:gpu serve \
                                        --production --triton-options model-control-mode=explicit \
                                        --triton-options load-model=onnx_mnist --triton-options load-model=torchscript_yolov5s

          .. tab-item:: CPU
              :sync: cpu

              .. code-block:: bash

                  $ docker run --init --rm -p 3000:3000 triton-integration:cpu serve-grpc \
                                        --production --triton-options model-control-mode=explicit \
                                        --triton-options load-model=onnx_mnist --triton-options load-model=torchscript_yolov5s

    .. tab-item:: Linux
       :sync: linux

       .. tab-set::

          .. tab-item:: GPU
              :sync: gpu

              .. code-block:: bash

                  $ docker run --init --rm --network=host triton-integration:gpu serve \
                                        --production --triton-options model-control-mode=explicit \
                                        --triton-options load-model=onnx_mnist --triton-options load-model=torchscript_yolov5s

          .. tab-item:: CPU
              :sync: cpu

              .. code-block:: bash

                  $ docker run --init --rm --network=host triton-integration:cpu serve-grpc \
                                        --production --triton-options model-control-mode=explicit \
                                        --triton-options load-model=onnx_mnist --triton-options load-model=torchscript_yolov5s

.. tip::

   To see all available options for Triton run:

   .. code-block:: bash

      $ docker run --init --rm -p 3000:3000 triton-integration:gpu tritonserver --help

Current caveats
~~~~~~~~~~~~~~~

Versioning policy limitations
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

By default, TritonRunner uses gRPC protocol that is defined in Inference protocol [#triton_inference_protocol]_.

HTTP/REST APIs is disabled by default, though it can be enabled via ``BENTOML_USE_HTTP_TRITONSERVER`` environment variable.

.. code-block:: bash

    $ docker run --init --rm -p 3000:3000 triton-integration:cpu serve \
                            --production --triton-options model-control-mode=explicit \
                            --triton-options load-model=onnx_mnist --triton-options load-model=torchscript_yolov5s

.. epigraph::
    Currently, TritonRunner does not support running `Metrics server <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html>`_.
    If you are interested in supporting the metrics server, please open an issue on :github:`GitHub <bentoml/BentoML/issues/new/choose>`

Additionally, BentoML supervisors will allocate a random port for the gRPC server, hence ``grpc-port`` options that is passed to ``--triton-options`` will be omited.

---------------

Remarks
~~~~~~~

It is worth to mention that Triton Inference Server is a highly optimized model server that aims simplify multi-models serving.
The following list of benefits and drawbacks are based on our experience building this integration:

:raw-html:`<br />`

The benefits Triton Inference Server brings:

* **Performance enhancement**: Triton Inference Server's concurrent model execution [#concurrent_model_execution]_
  enables models to be executed concurrently on the same/multiple GPUs, with out the IO performance limitations introduced by Python's GIL [#constraints]_.

* **Extensive configuration options**: To go along with its suite of features and optimizations, Triton Inference Server also provides
  an extensive set of configuration options that can be used to customize models' behaviour, with ability to extend its capabilities.

* **Framework supports and custom backends**: Triton supports all major frameworks, including Tensorflow, PyTorch, ONNX, TensorRT [#tensorrt_support]_ and OpenVINO,
  which enables for a wide range of use-cases and empowers ML practitioners to do what they do best. Additionally, they also support custom backends
  that can be used to implement custom inference logic.


Should I use Triton Inference Server?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Absolutely, depending on your use-case. If you are looking for improving RPS for your production models, then Triton Inference Server through ``bentoml.triton.Runner``
is a great option to consider. Note that you are responsible for managing the model repository and its respectively ``config.pbtxt`` to make sure it will fit your use-case.

If you have a lot of models to manage, then we recommend to only use TritonRunner for larger models that requires more resources to run inference, and use BentoML's
:ref:`Runner <concepts/runner:Using Runners>` for smaller models.

If you are a Triton users who are looking for a more easier and streamlined workflow for multi-model inference graphs and more ergonomic pre/post-processing, 
then you might want to consider this integration as a solution.

.. admonition:: 🚧 Help us improve the integration!

    This integration is still in its early stages and we are looking for feedbacks and contributions to make it even better!

    If you have any feedback or want to contribute any improvements to the
    Triton Inference Server integration, we would love to see your `feature requests <https://github.com/bentoml/BentoML/issues>`_
    and `pull request! <https://github.com/bentoml/BentoML/pulls>`_

    Check out the `BentoML development guide <https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md>`_
    and `documentation guide <https://github.com/bentoml/BentoML/blob/main/docs/README.md>`_
    to get started.

----

.. rubric:: Footnotes

.. [#triton] :github:`NVIDIA Triton Inference Server <triton-inference-server>` GitHub repository.

.. [#constraints] The given benchmark is tested with this :github:`code <bentoml/BentoML/tree/main/examples/triton_runner/locustfile.py>`, with 8 CPUs and 1 Tesla T4 GPU, under 50u/s to 500 concurrent users.

.. [#triton_docs] `Triton Inference Server documentation <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html>`_

.. [#triton_runner] The :github:`example project <bentoml/BentoML/tree/main/examples/triton>` includes code for both YOLOv5 model in Tensorflow, ONNX, and TorchScript.

.. [#triton_build] Building Triton: `[link] <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/build.html>`_

.. [#triton_inputs_outputs] Inputs/Outputs specified under ``config.pbtxt``: `[link] <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#inputs-and-outputs>`_

.. [#infer_output_tensor] InferOutputTensor protobuf message: `[link] <https://github.com/triton-inference-server/common/blob/7b37a244e939c69c51df74d31cd905ae6ffec2f9/protobuf/grpc_service.proto#L703>`_

.. [#model_infer_response] ModelInferResponse protobuf message: `[link] <https://github.com/triton-inference-server/common/blob/7b37a244e939c69c51df74d31cd905ae6ffec2f9/protobuf/grpc_service.proto#L696>`_

.. [#triton_inference_protocol] Inference protocol: `[link] <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/inference_protocols.html?highlight=inference%20api>`_

.. [#concurrent_model_execution] Concurrent model execution: `[link] <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#concurrent-model-execution>`_

.. [#tensorrt_support] https://www.bloomberg.com/press-releases/2021-11-09/nvidia-announces-major-updates-to-triton-inference-server-as-25-000-companies-worldwide-deploy-nvidia-ai-inference

.. [#triton_container_ngc] Triton Inference Server from NGC catalog: `[link] <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver>`_
