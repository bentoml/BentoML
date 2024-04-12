=============
GPU inference
=============

GPU inference with BentoML helps you accelerate the computational efficiency of machine learning models. BentoML provides a streamlined approach to deploying :doc:`/guides/services` that can leverage GPU resources for inference tasks.

This document explains how to configure and allocate GPUs to run inference with BentoML.

Configure GPU resources
-----------------------

When creating your BentoML Service, you need to make sure your Service implementation has the correct GPU configuration.

A single device
^^^^^^^^^^^^^^^

When a single GPU is available, frameworks like PyTorch and TensorFlow default to using ``cuda:0`` or ``cuda``.  In PyTorch, for example, to assign a model to use the GPU, you use ``.to('cuda:0')``. An example of setting up a BentoML Service to use the a single GPU:

.. code-block:: python

    @bentoml.service(resources={"gpu": 1})
    class MyService:
        def __init__(self):
            import torch
            self.model = torch.load('model.pth').to('cuda:0')

Multiple devices
^^^^^^^^^^^^^^^^

In systems with multiple GPUs, each GPU is assigned an index starting from 0 (``cuda:0``, ``cuda:1``, ``cuda:2``, etc.). You can specify which GPU to use or distribute operations across multiple GPUs.

To use a specific GPU:

.. code-block::

    @bentoml.service(resources={"gpu": 2})
    class MultiGPUService:
        def __init__(self):
            import torch
            self.model1 = torch.load('model1.pth').to("cuda:0")  # Using the first GPU
            self.model2 = torch.load('model2.pth').to("cuda:1")  # Using the second GPU

This image explains how different models use the GPUs assigned to them.

.. image:: ../../_static/img/guides/gpu-inference/gpu-inference-architecture.png
    :width: 400px
    :align: center

.. note::

    Workers are the processes that actually run the code logic within a BentoML Service. By default, a BentoML Service has one worker. It is possible to set multiple workers and allocate specific GPUs to individual workers. See :doc:`/guides/workers` for details.

If you want to use multiple GPUs for distributed operations (multiple GPUs for the same worker), PyTorch and TensorFlow offer different methods:

- PyTorch: `DataParallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html>`_ and `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_
- TensorFlow: `Distributed training <https://www.tensorflow.org/guide/distributed_training>`_

Deployment on BentoCloud
^^^^^^^^^^^^^^^^^^^^^^^^

When deploying on BentoCloud, specify ``resources`` with ``gpu`` or ``gpu_type`` in the ``@bentoml.service`` decorator to allow BentoCloud to allocate the necessary GPU resources:

.. code-block:: python

    @bentoml.service(
        resources={
            "gpu": 1, # The number of allocated GPUs
            "gpu_type": "nvidia-l4" # A specific GPU type on BentoCloud
        }
    )
    class MyService:
        # Service implementation

To list available GPU types on your BentoCloud account, run:

.. code-block:: bash

    $ bentoml deployment list-instance-types

    Name        Price  CPU    Memory  GPU  GPU Type
    cpu.1       *      500m   2Gi
    cpu.2       *      1000m  2Gi
    cpu.4       *      2000m  8Gi
    cpu.8       *      4000m  16Gi
    gpu.t4.1    *      2000m  8Gi     1    nvidia-tesla-t4
    gpu.l4.1    *      4000m  16Gi    1    nvidia-l4
    gpu.a100.1  *      6000m  43Gi    1    nvidia-tesla-a100

Limit GPU visibility
--------------------

By setting ``CUDA_VISIBLE_DEVICES`` to the IDs of the GPUs you want to use, you can limit BentoML to only use certain GPUs for your Service. GPU IDs are typically numbered starting from 0. For example:

- ``CUDA_VISIBLE_DEVICES=0`` makes only the first GPU visible.
- ``CUDA_VISIBLE_DEVICES=1,2`` makes the second and third GPUs visible.
