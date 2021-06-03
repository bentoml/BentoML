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
- NVIDIA GPU that has compute capability ``>=3.0`` (find yours `from NVIDIA <https://developer.nvidia.com/cuda-gpus>`_)


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

General workaround (Recommended)
""""""""""""""""""""""""""""""""
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

2. Preparing BentoService for GPU Serving
-----------------------------------------

.. seealso:: BentoML's `gallery <https://github.com/bentoml/gallery>`_ for more detailed use-case

We will provide examples on preparing ``BentoService`` for *PyTorch, Tensorflow, and ONNX* with GPU enabled.


.. warning::
    As of **0.13.0**, Multiple GPUs Inference is currently not supported. (However, it is within our future roadmap to provide support for such feature.)

Docker Images
^^^^^^^^^^^^^

Users have options to build their own customized docker images to serve with ``BentoService`` via ``@env(docker_base_images="")``.
Make sure that your custom docker images have Python and CUDA library in order to run with GPU.

BentoML also provides three `CUDA-enabled images <https://hub.docker.com/r/bentoml/model-server/tags?page=1&ordering=last_updated&name=gpu>`_ with CUDA 11.3 and CUDNN 8.2.0 ( refers to this `support matrix <https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html>`_ for CUDA and CUDNN version matching ).

Tensorflow
^^^^^^^^^^

.. note::
    If users want to utilize multiple GPUs while training, refers to Tensorflow's `distributed strategies <https://www.tensorflow.org/guide/distributed_training>`_

Luckily, Tensorflow code with ``tf.keras`` model will run transparently on a single GPU without any changes. One can read more `here <https://www.tensorflow.org/guide/gpu>`_.

.. warning::

    **NOT RECOMMEND** to manually set device placement unless you know what you are doing!

        During training, if one choose to manually set device placement for specific operations, e.g:

        .. code-block:: python

            tf.debugging.set_log_device_placement(True)

            # Place tensors on GPU
            # train my_model on GPU:0
            with tf.device("/GPU:0"):
                ...

        then make sure you correctly create your model during inference to avoid any potential errors.

        .. code-block:: python

            # my_model_gpu is a trained on GPU:0, with weight and tokenizer to file
            with tf.device("/GPU:0"):
                my_inference_model = build_model() # build_model
                my_inference_model.set_weights(my_model_gpu.get_weights())
                ... # continue with your inference tasks.

``BentoService`` definition with CUDA-enabled Images
""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: python

    # bento_svc.py
    from bentoml import BentoService, api, artifacts, env
    from bentoml.adapters import JsonInput
    from bentoml.frameworks.keras import KerasModelArtifact
    from bentoml.service.artifacts.common import PickleArtifact

    @env(pip_packages=['tensorflow', 'scikit-learn', 'pandas'] ,\
          docker_base_image="bentoml/model-server:0.12.1-py38-gpu")
    @artifacts([KerasModelArtifact('model'), PickleArtifact('tokenizer')])
    class TensorflowService(BentoService):
        def preprocessing(self, text_str):
            proc = text_to_word_sequence(preprocess(text_str))
            tokens = list(map(self.word_to_index, proc))
            return tokens

        @api(input=JsonInput())
        def predict(self, parsed_json):
            raw = self.preprocessing(parsed_json['text'])
            input_data = [raw[: n + 1] for n in range(len(raw))]
            input_data = pad_sequences(input_data, maxlen=100, padding="post")
            return self.artifacts.model.predict(input_data)

Bundle our BentoService
"""""""""""""""""""""""
.. code-block:: python

    # bento_packer.py
    from bento_service import TensorflowService

    config.experimental.set_memory_growth(gpu[0], True) # fallback options to remove memory limit

    def load_tokenizer():
        # load your saved tokenizer
        ...

    def load_model():
        # load tf model json and weights
        ...


    model = load_model()
    tokenizer = load_tokenizer()

    bento_svc = TensorflowService()
    bento_svc.pack('model', model)
    bento_svc.pack('tokenizer', tokenizer)

    saved_path = bento_svc.save()

Run Inference with our BentoService
"""""""""""""""""""""""""""""""""""

.. code-block:: bash

    $ bentoml serve TensorflowService:latest

    ...

.. code-block:: bash

    $ nvidia-smi
    Thu Jun  3 17:02:06 2021
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 465.31       Driver Version: 465.31       CUDA Version: 11.3     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
    | N/A   59C    P8     5W /  N/A |      6MiB /  6078MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A      1418      G   /opt/conda/venv/bin/python       5781MiB |
    +-----------------------------------------------------------------------------+