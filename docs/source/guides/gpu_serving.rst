==============================
GPU Serving with BentoML
==============================

It is widely recognized within the academia world and industry that GPUs have superior benefits over CPU-based platform due to its speed and efficiency advantages for both training and inference
tasks, as shown `by NVIDIA <https://www.nvidia.com/content/tegra/embedded-systems/pdf/jetson_tx1_whitepaper.pdf>`_.

Almost every deep learning frameworks (Tensorflow, PyTorch, ONNX, etc.) have supports for GPU, this guide demonstrates how to serve your ``BentoService`` with GPU.

1. Prerequisite
---------------

- ``GNU/Linux x86_64`` with kernel version ``>=3.10``. (``uname -a`` to check)
- Docker >=19.03
- NVIDIA GPU that has compute capability ``>=3.0`` (find out `here <https://developer.nvidia.com/cuda-gpus>`_)


NVIDIA Drivers
^^^^^^^^^^^^^^
Make sure you have installed NVIDIA driver for your Linux distribution. The recommended way to install drivers is to use the package manager of your distribution but other alternatives are also `available <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_.

For instruction on how to use your package manager to install drivers from CUDA network repository, follow this `guide <https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>`_

NVIDIA Container Toolkit
^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::
    NVIDIA provides `detailed instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_ for installing both ``Docker CE`` and ``nvidia-docker``

.. note::
    For Arch users install ``nvidia-docker`` via `AUR <https://aur.archlinux.org/packages/nvidia-docker/>`_

.. warning::
    Recent updates to ``systemd`` re-architecture, which is described via `#1447 <https://github.com/NVIDIA/nvidia-docker/issues/1447>`_, completely breaks ``nvidia-docker``.
    This issue is confirmed to be `patched <https://github.com/NVIDIA/nvidia-docker/issues/1447#issuecomment-760189260>`_ for future releases.

General workaround (recommend)
""""""""""""""""""""""""""""""
    Append device location to ``--device`` when running the container.

    .. code-block:: bash

        $ docker run --gpus all --device /dev/nvidia0 \
                       --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
                       --device /dev/nvidia-modeset --device /dev/nvidiactl <docker-args>

    If one choose to make use of Makefile, add the following:

    .. code-block::

    	DEVICE_ARGS := --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-modeset --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools

        # example docker run
        svc-d-r:
            docker run --gpus all $(DEVICE_ARGS) foo:bar

Debian-based
""""""""""""
    Disable ``cgroup`` hierarchy by adding to ``systemd.unified_cgroup_hierarchy=0`` to ``GRUB_CMDLINE_LINUX_DEFAULT``

    .. code-block::

        GRUB_CMDLINE_LINUX_DEFAULT="quiet systemd.unified_cgroup_hierarchy=0"

Arch-based
""""""""""
    For Arch users, change ``#no-cgroups=false`` to ``no-cgroups=true`` under ``/etc/nvidia-container-runtime/config.toml``

docker-compose
""""""""""""""
    For ```docker-compose`` added the following:

    .. code-block::

        # docker-compose.yaml

        ...
        devices:
          - /dev/nvidia0:/dev/nvidia0
          - /dev/nvidiactl:/dev/nvidiactl
          - /dev/nvidia-modeset:/dev/nvidia-modeset
          - /dev/nvidia-uvm:/dev/nvidia-uvm
          - /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools

2.
---------------