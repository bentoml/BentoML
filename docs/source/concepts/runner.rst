=============
Using Runners
=============

*time expected: 15 minutes*

This page articulates on the concept of Runners and demonstrates its role within the
BentoML architecture.

What is Runner?
---------------

In BentoML, Runner represents a unit of computation that can be executed on a remote
Python worker and scales independently.

Runner allows :ref:`bentoml.Service <reference/core:bentoml.Service>` to parallelize
multiple instances of a :ref:`bentoml.Runnable <reference/core:bentoml.Runnable>` class,
each on its own Python worker. When a BentoServer is launched, a group of runner worker
processes will be created, and :code:`run` method calls made from the
:code:`bentoml.Service` code will be scheduled among those runner workers.

Runner also supports :doc:`/guides/batching`. For a
:ref:`bentoml.Runnable <reference/core:bentoml.Runnable>` configured with batching,
multiple :code:`run` method invocations made from other processes can be dynamically
grouped into one batch execution in real-time. This is especially beneficial for compute
intensive workloads such as model inference, helps to bring better performance through
vectorization or multi-threading.


Pre-built Model Runners
-----------------------

BentoML provides pre-built Runners implemented for each ML framework supported. These
pre-built runners are carefully configured to work well with each specific ML framework.
They handle working with GPU when GPU is available, set the number of threads and number
of workers automatically, and convert the model signatures to corresponding Runnable
methods.

.. code:: python

    trained_model = train()

    bentoml.pytorch.save_model(
        "demo_mnist",  # model name in the local model store
        trained_model,  # model instance being saved
        signatures={   # model signatures for runner inference
            "predict": {
                "batchable": True,
                "batch_dim": 0,
            }
        }
    )

    runner = bentoml.pytorch.get("demo_mnist:latest").to_runner()
    runner.init_local()
    runner.predict.run( MODEL_INPUT )


.. _custom-runner:

Custom Runner
-------------

For more advanced use cases, BentoML also allows users to define their own Runner
classes. This is useful when the pre-built Runners do not meet the requirements, or
when the user wants to implement a Runner for a new ML framework.

Creating a Runnable
^^^^^^^^^^^^^^^^^^^

Runner can be created from a :ref:`bentoml.Runnable <reference/core:bentoml.Runnable>`
class. By implementing a :code:`Runnable` class, users can create Runner instances that
runs custom logic. Here's an example (excerpted from :examples:`one of our example projects
<custom_runner/nltk_pretrained_model>`)
creating an NLTK runner that does sentiment analysis with a pre-trained model:

.. literalinclude:: ../../../examples/custom_runner/nltk_pretrained_model/service.py
   :language: python
   :caption: `service.py`

.. note::

    Full code example can be found :examples:`here <custom_runner/nltk_pretrained_model>`.


The constant attribute ``SUPPORTED_RESOURCES`` indicates which resources this Runnable class
implementation supports. The only currently pre-defined resources are ``"cpu"`` and
``"nvidia.com/gpu"``.

The constant attribute ``SUPPORTS_CPU_MULTI_THREADING`` indicates whether or not the runner supports
CPU multi-threading.

.. tip::

    Neither constant can be set inside of the runner's ``__init__`` or ``__new__`` methods, as they are class-level attributes. The reason being BentoML’s scheduling policy is not invoked in runners’ initialization code, as instantiating runners can be quite expensive.

Since NLTK library doesn't support utilizing GPU or multiple CPU cores natively, supported resources
is specified as :code:`("cpu",)`, and ``SUPPORTS_CPU_MULTI_THREADING`` is set to False. This is the default configuration.
This information is then used by the BentoServer scheduler to determine the worker pool size for this runner.

The :code:`bentoml.Runnable.method` decorator is used for creating
:code:`RunnableMethod` - the decorated method will be exposed as the runner interface
for accessing remotely. :code:`RunnableMethod` can be configured with a signature,
which is defined same as the :ref:`concepts/model:Model Signatures`.

More examples about custom runners implementing their own :code:`Runnable` class can be found at:
:examples:`examples/custom_runner <custom_runner>`.

Reusable Runnable
^^^^^^^^^^^^^^^^^

Runnable class can also take :code:`__init__` parameters to customize its behavior for
different scenarios. The same Runnable class can also be used to create multiple runners
and used in the same service. For example:

.. code-block:: python
   :caption: `service.py`

    import bentoml
    import torch

    class MyModelRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu",)
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self, model_file):
            self.model = torch.load_model(model_file)

        @bentoml.Runnable.method(batchable=True, batch_dim=0)
        def predict(self, input_tensor):
            return self.model(input_tensor)

    my_runner_1 = bentoml.Runner(
        MyModelRunnable,
        name="my_runner_1",
        runnable_init_params={
            "model_file": "./saved_model_1.pt",
        }
    )
    my_runner_2 = bentoml.Runner(
        MyModelRunnable,
        name="my_runner_2",
        runnable_init_params={
            "model_file": "./saved_model_2.pt",
        }
    )

    svc = bentoml.Service(__name__, runners=[my_runner_1, my_runner_2])

.. epigraph::
    All runners presented in one ``bentoml.Service`` object must have unique names.

.. note::

    The default Runner name is the Runnable class name. When using the same Runnable
    class to create multiple runners and use them in the same service, user must rename
    runners by specifying the ``name`` parameter when creating the runners. Runner
    name are a key to configuring individual runner at deploy time and to runner related
    logging and tracing features.


Custom Model Runner
^^^^^^^^^^^^^^^^^^^

Custom Runnable built with Model from BentoML's model store:

.. code::

    from typing import Any

    import bentoml
    from bentoml.io import JSON
    from bentoml.io import NumpyNdarray
    from numpy.typing import NDArray

    bento_model = bentoml.pytorch.get("spam_detection:latest")

    class SpamDetectionRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("cpu",)
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            # load the model instance
            self.classifier = bentoml.sklearn.load_model(bento_model)

        @bentoml.Runnable.method(batchable=False)
        def is_spam(self, input_data: NDArray[Any]) -> NDArray[Any]:
            return self.classifier.predict(input_data)

    spam_detection_runner = bentoml.Runner(SpamDetectionRunnable, models=[bento_model])
    svc = bentoml.Service("spam_detector", runners=[spam_detection_runner])

    @svc.api(input=NumpyNdarray(), output=JSON())
    def analysis(input_text: NDArray[Any]) -> dict[str, Any]:
        return {"res": spam_detection_runner.is_spam.run(input_text)}


Custom Runnable can be also built by extending the Runnable class generated by Model
from BentoML's model store. A full example with custom metrics to monitor the model's
performance can be found at :examples:`examples/custom_model_runner <custom_model_runner>`.


Serving Multiple Models via Runner
----------------------------------

Serving multiple models in the same workflow is also a common pattern in BentoML’s prediction framework. This pattern can be achieved by simply instantiating multiple runners up front and passing them to the service that’s being created. Each runner/model will be configured with its’ own resources and run autonomously. If no configuration is passed, BentoML will then determine the optimal resources to allocate to each runner.


Sequential Runs
^^^^^^^^^^^^^^^

.. code:: python

    import asyncio
    import bentoml
    import PIL.Image

    import bentoml
    from bentoml.io import Image, Text

    transformers_runner = bentoml.transformers.get("sentiment_model:latest").to_runner()
    ocr_runner = bentoml.easyocr.get("ocr_model:latest").to_runner()

    svc = bentoml.Service("sentiment_analysis", runners=[transformers_runner, ocr_runner])

    @svc.api(input=Image(),output=Text())
    def classify(input: PIL.Image.Image) -> str:
        ocr_text = ocr_runner.run(input)
        return transformers_runner.run(ocr_text)

It’s as simple as creating two runners and invoking them synchronously in your prediction endpoint. Note that an async endpoint is often preferred in these use cases as the primary event loop is yielded while waiting for other IO-expensive tasks.

For example, the same API above can be achieved as an ``async`` endpoint:


.. code:: python

    @svc.api(input=Image(),output=Text())
    async def classify_async(input: PIL.Image.Image) -> str:
        ocr_text = await ocr_runner.async_run(input)
        return await transformers_runner.async_run(ocr_text)


Concurrent Runs
^^^^^^^^^^^^^^^

In cases where certain steps can be executed concurrently, :code:`asyncio.gather` can be used to aggregate results from multiple concurrent runs. For instance, if you are running two models simultaneously, you could invoke ``asyncio.gather`` as follows:

.. code-block:: python

    import asyncio
    import PIL.Image

    import bentoml
    from bentoml.io import Image, Text

    preprocess_runner = bentoml.Runner(MyPreprocessRunnable)
    model_a_runner = bentoml.xgboost.get('model_a:latest').to_runner()
    model_b_runner = bentoml.pytorch.get('model_b:latest').to_runner()

    svc = bentoml.Service('inference_graph_demo', runners=[
        preprocess_runner,
        model_a_runner,
        model_b_runner
    ])

    @svc.api(input=Image(), output=Text())
    async def predict(input_image: PIL.Image.Image) -> str:
        model_input = await preprocess_runner.async_run(input_image)

        results = await asyncio.gather(
            model_a_runner.async_run(model_input),
            model_b_runner.async_run(model_input),
        )

        return post_process(
            results[0], # model a result
            results[1], # model b result
        )


Once each model completes, the results can be compared and logged as a post processing
step.

Another example generating text using three different language models concurrently and
then classify each generated paragraph with an classification model can be found at:
:examples:`examples/inference_graph <inference_graph>`.


Runner Definition
-----------------

.. TODO::
    Document detailed list of Runner options

    .. code:: python

        my_runner = bentoml.Runner(
            MyRunnable,
            runnable_init_params={"foo": foo, "bar": bar},
            name="custom_runner_name",
            strategy=None, # default strategy will be selected depending on the SUPPORTED_RESOURCES and SUPPORTS_CPU_MULTI_THREADING flag on runnable
            models=[..],

            # below are also configurable via config file:

            # default configs:
            max_batch_size=..  # default max batch size will be applied to all run methods, unless override in the runnable_method_configs
            max_latency_ms=.. # default max latency will be applied to all run methods, unless override in the runnable_method_configs

            runnable_method_configs=[
                {
                    method_name="predict",
                    max_batch_size=..,
                    max_latency_ms=..,
                }
            ],
        )

Runner Configuration
--------------------

Runner behaviors and resource allocation can be specified via BentoML :ref:`configuration <guides/configuration:Configuration>`.

Runners can be both configured individually or in aggregate under the ``runners`` configuration key. To configure a specific runner, specify its name
under the ``runners`` configuration key. Otherwise, the configuration will be applied to all runners. The examples below demonstrate both
the configuration for all runners in aggregate and for an individual runner (``iris_clf``).

Adaptive Batching
^^^^^^^^^^^^^^^^^

If a model or custom runner supports batching, the :ref:`adaptive batching <guides/batching:Adaptive Batching>` mechanism is enabled by default.
To explicitly disable or control adaptive batching behaviors at runtime, configuration can be specified under the ``batching`` key.

.. tab-set::

    .. tab-item:: All Runners
       :sync: all_runners

       .. code-block:: yaml
          :caption: ⚙️ `configuration.yml`

          runners:
            batching:
              enabled: true
              max_batch_size: 100
              max_latency_ms: 500

    .. tab-item:: Individual Runner
        :sync: individual_runner

        .. code-block:: yaml
           :caption: ⚙️ `configuration.yml`

           runners:
             iris_clf:
               batching:
                 enabled: true
                 max_batch_size: 100
                 max_latency_ms: 500

Resource Allocation
^^^^^^^^^^^^^^^^^^^

By default, a runner will attempt to utilize all available resources in the container. Runner's resource allocation can also be customized
through configuration, with a `float` value for ``cpu`` and an `int` value for ``nvidia.com/gpu``. Fractional GPU is currently not supported.

.. tab-set::

    .. tab-item:: All Runners
       :sync: all_runners

       .. code-block:: yaml
          :caption: ⚙️ `configuration.yml`

          runners:
            resources:
              cpu: 0.5
              nvidia.com/gpu: 1

    .. tab-item:: Individual Runner
        :sync: individual_runner

        .. code-block:: yaml
           :caption: ⚙️ `configuration.yml`

           runners:
             iris_clf:
               resources:
                 cpu: 0.5
                 nvidia.com/gpu: 1

Alternatively, a runner can be mapped to a specific set of GPUs. To specify GPU mapping, instead of defining an `integer` value, a list of device IDs
can be specified for the ``nvidia.com/gpu`` key. For example, the following configuration maps the configured runners to GPU device 2 and 4.

.. tab-set::

    .. tab-item:: All Runners
       :sync: all_runners

       .. code-block:: yaml
          :caption: ⚙️ `configuration.yml`

          runners:
            resources:
              nvidia.com/gpu: [2, 4]

    .. tab-item:: Individual Runner
       :sync: individual_runner

       .. code-block:: yaml
          :caption: ⚙️ `configuration.yml`

          runners:
            iris_clf:
              resources:
                nvidia.com/gpu: [2, 4]

For the detailed information on the meaning of each resource allocation configuration, see :doc:`/guides/scheduling`.

Traffic Control
^^^^^^^^^^^^^^^

Same as API server, you can also configure the traffic settings for both all runners and individual runner.
Specifcally, ``traffic.timeout`` defines the amount of time in seconds that the runner will wait for a response from the model before timing out.
``traffic.max_concurrency`` defines the maximum number of concurrent requests the runner will accept before returning an error.

.. tab-set::

    .. tab-item:: All Runners
       :sync: all_runners

       .. code-block:: yaml
          :caption: ⚙️ `configuration.yml`

          runners:
            traffic:
              timeout: 60
              max_concurrency: 10

    .. tab-item:: Individual Runner
       :sync: individual_runner

       .. code-block:: yaml
          :caption: ⚙️ `configuration.yml`

          runners:
            iris_clf:
              traffic:
                timeout: 60
                max_concurrency: 10

Access Logging
^^^^^^^^^^^^^^

See :ref:`guides/logging:Logging Configuration` for access log customization.


Distributed Runner with Yatai
-----------------------------

`🦄️ Yatai <https://github.com/bentoml/Yatai>`_ provides a more advanced Runner
architecture specifically designed for running large scale inference workloads on a
Kubernetes cluster.

While the standalone :code:`BentoServer` schedules Runner workers on their own Python
processes, the :code:`BentoDeployment` created by Yatai, scales Runner workers in their
own group of `Pods <https://kubernetes.io/docs/concepts/workloads/pods/>`_ and made it
possible to set a different resource requirement for each Runner, and auto-scaling each
Runner separately based on their workloads.


Sample :code:`BentoDeployment` definition file for deploying in Kubernetes:

.. code:: yaml

    apiVersion: yatai.bentoml.org/v1beta1
    kind: BentoDeployment
    spec:
    bento_tag: 'fraud_detector:dpijemevl6nlhlg6'
    autoscaling:
        minReplicas: 3
        maxReplicas: 20
    resources:
        limits:
            cpu: 500m
        requests:
            cpu: 200m
    runners:
    - name: model_runner_a
        autoscaling:
            minReplicas: 1
            maxReplicas: 5
        resources:
            requests:
                nvidia.com/gpu: 1
                cpu: 2000m
            ...

.. TODO::
    add graph explaining Yatai Runner architecture
