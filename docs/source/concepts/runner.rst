=============
Using Runners
=============

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


Custom Runner
-------------

Creating a Runnable
^^^^^^^^^^^^^^^^^^^

Runner can be created from a :ref:`bentoml.Runnable <reference/core:bentoml.Runnable>`
class. By implementing a :code:`Runnable` class, users can create Runner instances that
runs custom logic. Here's an example, creating an NLTK runner that does sentiment
analysis with a pre-trained model:

.. code:: python

    # service.py
    import bentoml
    import nltk
    from bentoml.io import Text, JSON
    from nltk.sentiment import SentimentIntensityAnalyzer
    from statistics import mean


    class NLTKSentimentAnalysisRunnable(bentoml.Runnable):

        SUPPORT_NVIDIA_GPU = False
        SUPPORT_CPU_MULTI_THREADING = False

        def __init__(self):
            nltk.download('vader_lexicon')
            nltk.download('punkt')
            self.sia = SentimentIntensityAnalyzer()

        @bentoml.Runnable.method(batchable=False)
        def is_positive(self, input_text):
            scores = [
                self.sia.polarity_scores(sentence)["compound"]
                for sentence in nltk.sent_tokenize(input_text)
            ]
            return mean(scores) > 0

    nltk_runner = bentoml.Runner(NLTKSentimentAnalysisRunnable)

    svc = bentoml.Service('sentiment_analyzer', runners=[nltk_runner])

    @svc.api(input=Text(), output=JSON())
    def analysis(input_text):
        is_positive = nltk_runner.is_positive.run(input_text)
        return { "is_positive": is_positive }

Run the service:

.. code:: bash

    bentoml serve service.py:svc

Send a test request:

.. code:: bash

    curl -X POST -H "content-type: application/text" --data "BentoML is great" http://127.0.0.1:3000/analysis

    {"is_positive":true}%


The :code:`SUPPORT_NVIDIA_GPU` and :code:`SUPPORT_CPU_MULTI_THREADING` class attribute indicates
whether or not this Runnable class implementation supports using GPU or utilizes multi
threading. Since NLTK library doesn't support utilizing GPU nor multi-cpu-core natively,
they are both set to False. This information is used by the BentoServer scheduler
to determine the worker pool size of this runner.

The :code:`bentoml.Runnable.method` decorator is used for creating
:code:`RunnableMethod` - the decorated method will be exposed as the runner interface
for accessing remotely. :code:`RunnableMethod` can be configured with a signature,
which is defined same as the :ref:`concepts/model:Model Signatures`.


Reusable Runnable
^^^^^^^^^^^^^^^^^

Runnable class can also take :code:`__init__` parameters to customize its behavior for
different scenarios. The same Runnable class can also be used to create multiple runners
and used in the same service. For example:

.. code:: python

    import bentoml
    import torch

    class MyModelRunnable(bentoml.Runnable):
        SUPPORT_NVIDIA_GPU = True
        SUPPORT_CPU_MULTI_THREADING = True

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

.. note::
    All runners presented in one :code:`bentoml.Service` object must have unique names.
    The default Runner name is the Runnable class name. When using the same Runnable
    class to create multiple runners and use them in the same service, user must rename
    runners by specifying the :code:`name` parameter when creating the runners. Runner
    name are a key to configuring individual runner at deploy time and to runner related
    logging and tracing features.

.. TODO::
    Add example Runnable implementation with a batchable method

Custom Model Runner
^^^^^^^^^^^^^^^^^^^

.. TODO::
    Document creating custom Runnable with models from BentoML model store

.. code::

    import bentoml
    import torch

    bento_model = bentoml.pytorch.get("fraud_detect:latest")

    class MyPytorchRunnable(bentoml.Runnable):
        SUPPORT_NVIDIA_GPU = False
        SUPPORT_CPU_MULTI_THREADING = True

        def __init__(self):
            self.model = torch.load_model(bento_model.path)

        @bentoml.Runnable.method(
            batchable=True,
            batch_dim=0,
        )
        def predict(self, input_tensor):
            return self.model(input_tensor)

    my_runner = bentoml.Runner(MyPytorchRunnable, models=[bento_model])


Runner Options
--------------

.. TODO::
    Document detailed list of Runner options

.. code:: python

    my_runner = bentoml.Runner(
        MyRunnable,
        runnable_init_params={"foo": foo, "bar": bar},
        name="custom_runner_name",
        strategy=None, # default strategy will be selected depending on the SUPPORT_NVIDIA_GPU and SUPPORT_CPU_MULTI_THREADING flag on runnable
        models=[..],

        # below are also configurable via config file:

        # default configs:
        cpu=4,
        nvidia_gpu=1
        custom_resources={..} # reserved API for supporting custom accelerators, a custom scheduling strategy will be needed to support new hardware types
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


Specifying Required Resources
-----------------------------

.. TODO::
    Document Runner resource specification, how it works, and how to override it with
    runtime configuration

.. code:: python

    my_runner = bentoml.Runner(MyRunnable, cpu=1)

    my_model_runner = bentoml.pytorch.get("my_model:latest").to_runner(gpu=1)


.. code:: yaml

    runners:
      - name: iris_clf
        cpu: 4
        nvidia_gpu: 0  # requesting 0 GPU
        max_batch_size: 20
      - name: my_custom_runner
        cpu: 2
        nvidia_gpu: 2  # requesting 2 GPUs
        runnable_method_configs:
          - name: "predict"
            max_batch_size: 10
            max_latency_ms: 500


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
