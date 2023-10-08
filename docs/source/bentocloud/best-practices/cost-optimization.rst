=================
Cost optimization
=================

This document contains a list of best practices for optimizing costs on BentoCloud. You can use the information on this page as
a reference when trying to reduce or rightsize your overall cloud spending. This helps you maximize the value of your cloud resources.

.. note::

    If you're new to BentoCloud, this document may not be the best place to start. We recommend you begin with this :doc:`/bentocloud/getting-started/quickstart`
    to understand the basics of BentoCloud.

Basic
-----

For those with a basic understanding of BentoML and BentoCloud, consider the following practices:

* **Local development and testing**. Develop and fully test your AI application locally until you think it is ready for deployment on BentoCloud. BentoML allows you to serve your model locally in the following ways.

  * Run ``bentoml serve service:svc --reload`` to test your BentoML Service.
  * After you package your model and all the necessary files into a Bento, run ``bentoml serve BENTO_TAG`` to test the Bento.
    Alternatively, use the ``bentoml.Server`` API if you are working with scripting environments or running Python-based tests.
  * Containerize the Bento and test the resulting Docker image using ``docker runs``.

  These practices help avoid deploying under-optimized models or services to the cloud, which may incur unnecessary costs from iterative fixing and redeploying.

* **Scale-to-zero**. Some services don't require constant availability and running them continuously is a cost drain. For those services,
  you can enable scale-to-zero by setting the minimum number of replicas allowed for scaling to 0. This way, you only pay for what you use, cutting costs during idle time.
  Note that for latency-insensitive services (for example, on-demand long-running inference jobs), we recommend you always enable scale-to-zero.

Advanced
--------

For seasoned users familiar with advanced use cases of BentoML, consider the following strategies for cost efficiency:

* **Benchmarking**. Benchmark your service in terms of throughput and latency across various hardware configurations (especially GPU types) and deployment settings.
  This helps you identify the most cost-effective setup and avoid overpaying or underutilizing resources.
* **Model parallelism**. Understand how parallelized your model can be. Efficiently parallelized models utilize resources better, reducing the need for more expensive compute power.
  If your model can be highly-parallelized, we recommend you enable :doc:`adaptive batching </guides/batching>` to send inputs to your model.
* **Distributing tasks on Runners**. BentoML :doc:`Runners </concepts/runner>` are the computation unit that can be executed on remote Python workers
  and scaled independently. Distributing tasks on Runners allows for more efficient resource usage, ensuring each runner is fully utilized without being overloaded. Specifically, you can do the following:

  * Relocate compute-heavy tasks (for example, model inference and heavy pre-processing) to dedicated Runners.
  * Allocate different models or stages of a pipeline to separate Runners for better resource management.

* **Adaptive scaling**. Take into account of a wide range of factors when setting your scaling strategies. Specifically, think about:

  * **Traffic**. Configure your scaling strategy based on observed and predicted traffic patterns. Dynamically adjusting resources based on demand
    prevents both over-provisioning (which costs more) and under-provisioning (which can degrade performance), thus ensuring resources are aligned with the actual demand.
  * **Resource and latency requirements**. Consider how much resources will be used over what time frame, and then set metric-based policies. For example, you have a GPU-bound application
    and you want to scale it up faster for lower latency. To do so, you can create a scaling policy based on GPU utilization metrics.
  * **Models**. For multi-model applications, we recommend you configure the scaling policy separately for your models, depending on how they are scheduled in your application.

* **GPU rightsizing**. GPUs are expensive. Using an ill-fitting GPU - either overpowered or underpowered - can lead to increased costs, so it is a good practice to rightsize your GPU based on your model.

  * For small models, using less powerful GPUs may not achieve the best performance, but as least you may have the desired performance.
  * Don't use expensive GPUs if you cannot make use of extra memory. Also note that when deploying large language models (LLMs), extra memory can be helpful for increasing throughput, via paged attention and continuous batching support in OpenLLM.

* **Exploring alternative runtimes**. Different runtimes offer varied performance benefits. Using the right runtime can improve efficiency, leading to faster results with fewer resources.
  Here are some of the runtimes supported by BentoML. You can assess their performance and cost implications to select the best fit for your AI application.

  * `FasterTransformer <https://github.com/NVIDIA/FasterTransformer>`_
  * `Torch compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_ and `TorchScript <https://pytorch.org/docs/stable/jit.html>`_
  * `ONNX Runtime <https://github.com/microsoft/onnxruntime>`_
  * `TensorRT <https://github.com/NVIDIA/TensorRT>`_ and `Triton <https://github.com/triton-inference-server/backend>`_
