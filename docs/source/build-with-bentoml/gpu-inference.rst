==============
Work with GPUs
==============

BentoML provides a streamlined approach to deploying :doc:`Services </build-with-bentoml/services>` that require GPU resources for inference tasks.

This document explains how to configure and allocate GPUs to run inference with BentoML.

Configure GPU resources
-----------------------

When creating your BentoML Service, you need to make sure your Service implementation has the correct GPU configuration.

A single device
^^^^^^^^^^^^^^^

When a single GPU is available, frameworks like PyTorch and TensorFlow default to using ``cuda:0`` or ``cuda``.  In PyTorch, for example, to assign a model to use the GPU, you use ``.to('cuda:0')``. An example of setting up a BentoML Service to use a single GPU:

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

.. image:: ../../_static/img/build-with-bentoml/gpu-inference/gpu-inference-architecture.png
    :width: 400px
    :align: center
    :alt: BentoML GPU inference architecture

.. note::

    Workers are the processes that actually run the code logic within a BentoML Service. By default, a BentoML Service has one worker. It is possible to set multiple workers and allocate specific GPUs to individual workers. See :doc:`/build-with-bentoml/parallelize-requests` for details.

If you want to use multiple GPUs for distributed operations (multiple GPUs for the same worker), PyTorch and TensorFlow offer different methods:

- PyTorch: `DataParallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html>`_ and `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_
- TensorFlow: `Distributed training <https://www.tensorflow.org/guide/distributed_training>`_

GPU deployment
--------------

When using PyTorch or TensorFlow to run models on GPUs, we recommend you directly install them along with their respective CUDA dependencies, via ``pip``. This ensures:

- **Minimal package size** since only the required components are installed.
- **Better compatibility** as the correct CUDA version is automatically installed alongside the frameworks.

For development, to install PyTorch or TensorFlow with the appropriate CUDA version using ``pip``, use the following commands:

.. code-block:: bash

    pip install torch
    pip install tensorflow[and-cuda]

When building your Bento, you DO NOT need to specify ``cuda_version`` again in your ``bentofile.yaml`` to install the CUDA toolkit separately. Simply add PyTorch and TensorFlow under ``packages`` (or they are in the separate ``requirements.txt`` file).

.. code-block:: yaml

    python:
      packages:
        - torch
        - tensorflow[and-cuda]

If you want to customize the installation of CUDA driver and libraries, use ``system_packages``, ``setup_script``, or ``base_image`` options under the :ref:`docker-configuration` field.

BentoCloud
^^^^^^^^^^

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

After your Service is ready, you can then deploy it to BentoCloud by running ``bentoml deploy .``. See :doc:`/scale-with-bentocloud/deployment/create-deployments` for details.

Docker
^^^^^^

You need to install the NVIDIA Container Toolkit for running Docker containers with Nvidia GPUs. NVIDIA provides `detailed instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_ for installing both ``Docker CE`` and ``nvidia-docker``.

After you build a Docker image for your Bento with ``bentoml containerize``, you can run it on all available GPUs like this:

.. code-block:: bash

    docker run --gpus all -p 3000:3000 bento_image:latest

You can use the ``device`` option to specify GPUs:

.. code-block:: bash

    docker run --gpus all --device /dev/nvidia0 \
                --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
                --device /dev/nvidia-modeset --device /dev/nvidiactl <docker-args>

To view GPU usage, use the ``nvidia-smi`` tool to see if a BentoML Service or Bento is using GPU. You can run it in a separate terminal while your BentoML Service is handling requests.

.. code-block:: bash

    # Refresh the output every second
    watch -n 1 nvidia-smi

Example output:

.. code-block:: bash

    Every 1.0s: nvidia-smi                            ps49pl48tek0: Mon Jun 17 13:09:46 2024

    Mon Jun 17 13:09:46 2024
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  NVIDIA A100-SXM4-80GB          On  | 00000000:00:05.0 Off |                    0 |
    | N/A   30C    P0              60W / 400W |   3493MiB / 81920MiB |      0%      Default |
    |                                         |                      |             Disabled |
    +-----------------------------------------+----------------------+----------------------+

    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |    0   N/A  N/A      1813      G   /usr/lib/xorg/Xorg                           70MiB |
    |    0   N/A  N/A      1946      G   /usr/bin/gnome-shell                         78MiB |
    |    0   N/A  N/A     11197      C   /Home/Documents/BentoML/demo/bin/python     3328MiB |
    +---------------------------------------------------------------------------------------+

For more information, see `the Docker documentation <https://docs.docker.com/config/containers/resource_constraints/#gpu>`_.

Limit GPU visibility
--------------------

By setting ``CUDA_VISIBLE_DEVICES`` to the IDs of the GPUs you want to use, you can limit BentoML to only use certain GPUs for your Service. GPU IDs are typically numbered starting from 0. For example:

- ``CUDA_VISIBLE_DEVICES=0`` makes only the first GPU visible.
- ``CUDA_VISIBLE_DEVICES=1,2`` makes the second and third GPUs visible.
