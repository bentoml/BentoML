================
Serving with GPU
================

Most popular deep learning frameworks (TensorFlow, PyTorch, ONNX, etc.) have supports
for GPU, both for training and inference. This guide demonstrates how to serve models
with BentoML on GPU.


Docker Images Options
---------------------

See :ref:`concepts/bento:Docker Options` for all options related to setting up docker
image options related to GPU. Here's a sample :code:`bentofile.yaml` config for serving
with GPU:

.. code:: yaml

    service: "service:svc"
    include:
    - "*.py"
    python:
        packages:
        - torch
        - torchvision
        - torchaudio
        extra_index_url:
        - "https://download.pytorch.org/whl/cu113"
    docker:
        distro: debian
        python_version: "3.8.12"
        cuda_version: "11.6.2"

When containerize a saved bento with a :code:`cuda_version` configured, BentoML will
install the corresponding cuda version onto the docker image created:

.. code-block:: bash

    $ bentoml containerize MyTFService:latest -t tf_svc

If the desired :code:`cuda_version` is not natively supported by BentoML, users can
still customize the installation of cuda driver and libraries via the
:code:`system_packages`, :code:`setup_script`, or :code:`base_image` options under the
:ref:`Bento build docker options<concepts/bento:Docker Options>`.


Running Docker with GPU
-----------------------

The NVIDIA Container Toolkit is required for running docker containers with Nvidia GPU.
NVIDIA provides `detailed instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_
for installing both :code:`Docker CE` and :code:`nvidia-docker`.

Start bento generated image and check for GPU usages:

.. code-block:: bash

    $ docker run --gpus all ${DEVICE_ARGS} -p 3000:3000 tf_svc:latest --workers=2

.. seealso::
    For more information, check out the `nvidia-docker wiki <https://github.com/NVIDIA/nvidia-docker/wiki>`_.


.. note::
    It is recommended to append device location to ``--device`` when running the
    container:

    .. code-block:: bash

        $ docker run --gpus all --device /dev/nvidia0 \
                       --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
                       --device /dev/nvidia-modeset --device /dev/nvidiactl <docker-args>


.. tip::
    In order to check for GPU usage, one can run ``nvidia-smi`` to check whether BentoService is using GPU. e.g

    .. code:: bash

        » nvidia-smi
        Thu Jun 10 15:30:28 2021
        +-----------------------------------------------------------------------------+
        | NVIDIA-SMI 465.31       Driver Version: 465.31       CUDA Version: 11.3     |
        |-------------------------------+----------------------+----------------------+
        | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
        |                               |                      |               MIG M. |
        |===============================+======================+======================|
        |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
        | N/A   49C    P8     6W /  N/A |    753MiB /  6078MiB |      0%      Default |
        |                               |                      |                  N/A |
        +-------------------------------+----------------------+----------------------+

        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A    179346      C   /opt/conda/bin/python             745MiB |
        +-----------------------------------------------------------------------------+
