================
Serving with GPU
================

Most popular deep learning frameworks (Tensorflow, PyTorch, ONNX, etc.) have supports
for GPU, both for training and inference. This guide demonstrates how to serve models
with BentoML on GPU.


NVIDIA Drivers
^^^^^^^^^^^^^^
Make sure you have installed NVIDIA driver for your Linux distribution. The recommended way to install drivers is to use the package manager of your distribution but other alternatives are also `available <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_.

For instruction on how to use your package manager to install drivers from CUDA network repository, follow this `guide <https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>`_.

NVIDIA Container Toolkit
^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::
    NVIDIA provides `detailed instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_ for installing both ``Docker CE`` and ``nvidia-docker``.
    Refers to ``nvidia-docker`` `wiki <https://github.com/NVIDIA/nvidia-docker/wiki>`_ for more information.

.. note::
    For Arch users install ``nvidia-docker`` via `AUR <https://aur.archlinux.org/packages/nvidia-docker/>`_.

.. warning::
    Recent updates to ``systemd`` re-architecture, which is described via `#1447 <https://github.com/NVIDIA/nvidia-docker/issues/1447>`_, completely breaks ``nvidia-docker``.
    This issue is confirmed to be `patched <https://github.com/NVIDIA/nvidia-docker/issues/1447#issuecomment-760189260>`_ for future releases.


General workaround (Recommended)
""""""""""""""""""""""""""""""""
    Append device location to ``--device`` when running the container.

    .. code-block:: bash

        $ docker run --gpus all --device /dev/nvidia0 \
                       --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
                       --device /dev/nvidia-modeset --device /dev/nvidiactl <docker-args>

    If one chooses to make use of ``Makefile`` then adds the following:

    .. code-block::

    	DEVICE_ARGS := --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-modeset --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools

        # example docker run
        svc-d-r:
            docker run --gpus all $(DEVICE_ARGS) foo:bar



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




Docker Images Options
^^^^^^^^^^^^^^^^^^^^^

Users have options to build their own customized docker images to serve with ``BentoService`` via ``@env(docker_base_images="")``.
Make sure that your custom docker images have Python and CUDA library in order to run with GPU.

BentoML also provides three `CUDA-enabled images <https://hub.docker.com/r/bentoml/model-server/tags?page=1&ordering=last_updated&name=gpu>`_
with CUDA 11.3 and CUDNN 8.2.0 (refers to this `support matrix <https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html>`_ for CUDA and CUDNN version matching).


.. code-block:: bash

    # to serve bento locally
    $ bentoml serve TensorflowService:latest

.. code-block:: bash

    # containerize saved bento
    $ bentoml containerize TensorflowService:latest -t tf_svc

.. code-block:: bash

    # start bento generated image and check for GPU usages:
    $ docker run --gpus all ${DEVICE_ARGS} -p 3000:3000 tf_svc:latest --workers=2
