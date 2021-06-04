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


1.1 NVIDIA Drivers
^^^^^^^^^^^^^^^^^^
Make sure you have installed NVIDIA driver for your Linux distribution. The recommended way to install drivers is to use the package manager of your distribution but other alternatives are also `available <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_.

For instruction on how to use your package manager to install drivers from CUDA network repository, follow this `guide <https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>`_.

1.2 NVIDIA Container Toolkit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::
    NVIDIA provides `detailed instructions <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_ for installing both ``Docker CE`` and ``nvidia-docker``.
    Refers to ``nvidia-docker`` `wiki <https://github.com/NVIDIA/nvidia-docker/wiki>`_ for more information.

.. note::
    For Arch users install ``nvidia-docker`` via `AUR <https://aur.archlinux.org/packages/nvidia-docker/>`_.

.. warning::
    Recent updates to ``systemd`` re-architecture, which is described via `#1447 <https://github.com/NVIDIA/nvidia-docker/issues/1447>`_, completely breaks ``nvidia-docker``.
    This issue is confirmed to be `patched <https://github.com/NVIDIA/nvidia-docker/issues/1447#issuecomment-760189260>`_ for future releases.

.. _general-workaround:

General workaround (Recommended)
""""""""""""""""""""""""""""""""
    Append device location to ``--device`` when running the container.

    .. code-block:: bash

        $ docker run --gpus all --device /dev/nvidia0 \
                       --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
                       --device /dev/nvidia-modeset --device /dev/nvidiactl <docker-args>

    If one choose to make use of :code:`Makefile`_ then add the following:

    .. code-block::

    	DEVICE_ARGS := --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-modeset --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools

        # example docker run
        svc-d-r:
            docker run --gpus all $(DEVICE_ARGS) foo:bar

Debian-based OS
"""""""""""""""
    Disable ``cgroup`` hierarchy by adding to ``systemd.unified_cgroup_hierarchy=0`` to ``GRUB_CMDLINE_LINUX_DEFAULT``.

    .. code-block::

        GRUB_CMDLINE_LINUX_DEFAULT="quiet systemd.unified_cgroup_hierarchy=0"

Others OS
"""""""""
    Change ``#no-cgroups=false`` to ``no-cgroups=true`` under ``/etc/nvidia-container-runtime/config.toml``.

docker-compose
""""""""""""""
    Added the following:

    .. code-block::

        # docker-compose.yaml

        ...
        devices:
          - /dev/nvidia0:/dev/nvidia0
          - /dev/nvidiactl:/dev/nvidiactl
          - /dev/nvidia-modeset:/dev/nvidia-modeset
          - /dev/nvidia-uvm:/dev/nvidia-uvm
          - /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools

2. Framework Support for GPU Inference with Implementation
----------------------------------------------------------

Jump to :ref:`tensorflow-impl` | :ref:`pytorch-impl` | :ref:`onnx-impl`


.. note::
    The examples we show here are merely demonstration on how GPU inference works among different frameworks to avoid bloating the guide.

.. seealso:: Please refers to BentoML's `gallery <https://github.com/bentoml/gallery>`_ for more detailed use-case on GPU Serving.

2.1 Preface
^^^^^^^^^^^

.. warning::
    As of **0.13.0**, Multiple GPUs Inference is currently not supported. (However, it is within our future roadmap to provide support for such feature)

.. note::
    In order to check for GPU usage, one can run ``nvidia-smi`` to check whether BentoService is using GPU. e.g

    .. code-block:: bash

        # BentoService is running in another session
        $ nvidia-smi
        Thu Jun  3 17:02:06 2021
        +-----------------------------------------------------------------------------+
        | NVIDIA-SMI 465.31       Driver Version: 465.31       CUDA Version: 11.3          |
        |-------------------------------+----------------------+----------------------+
        | GPU  Name        Persistence-M  | Bus-Id        Disp.A    | Volatile Uncorr. ECC |
        | Fan  Temp  Perf  Pwr:Usage/Cap |         Memory-Usage    | GPU-Util  Compute M. |
        |                                     |                           |               MIG M.    |
        |===============================+======================+======================|
        |   0  NVIDIA GeForce ...  Off    | 00000000:01:00.0 Off  |                  N/A     |
        | N/A   59C    P8    5W /  N/A     |      6MiB /  6078MiB   |      0%      Default    |
        |                                     |                          |                  N/A     |
        +-------------------------------+----------------------+----------------------+
        +-----------------------------------------------------------------------------+
        | Processes:                                                                                |
        |  GPU   GI   CI        PID   Type   Process name                       GPU Memory     |
        |        ID   ID                                                            Usage           |
        |=============================================================================|
        |    0   N/A  N/A      1418      G   /opt/conda/venv/bin/python       5781MiB        |
        +-----------------------------------------------------------------------------+

.. note::
    After each implementation:

    .. code-block:: bash

        # to serve our service locally
        $ bentoml serve TensorflowService:latest

    .. code-block:: bash

        # containerize our saved service
        $ bentoml containerize TensorflowService:latest -t tf_svc

    .. code-block:: bash

        # Start our container and check for GPU usages:
        $ docker run --gpus all ${DEVICE_ARGS} -p 5000:5000 tf_svc:latest --workers=2

.. note::
    see :ref:`general-workaround` for ``$DEVICE_ARGS``.


2.2 Docker Images Options
^^^^^^^^^^^^^^^^^^^^^^^^^

Users have options to build their own customized docker images to serve with ``BentoService`` via ``@env(docker_base_images="")``.
Make sure that your custom docker images have Python and CUDA library in order to run with GPU.

BentoML also provides three `CUDA-enabled images <https://hub.docker.com/r/bentoml/model-server/tags?page=1&ordering=last_updated&name=gpu>`_
with CUDA 11.3 and CUDNN 8.2.0 (refers to this `support matrix <https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html>`_ for CUDA and CUDNN version matching).

2.3 Tensorflow
^^^^^^^^^^^^^^

.. note::
    If users want to utilize multiple GPUs while training, refers to Tensorflow's `distributed strategies <https://www.tensorflow.org/guide/distributed_training>`_.

TLDR, Tensorflow code with ``tf.keras`` model will run transparently on a single GPU without any changes. One can read more `here <https://www.tensorflow.org/guide/gpu>`_.

.. warning::

    **NOT RECOMMEND** to manually set device placement unless you know what you are doing!

        During training, if one choose to manually set device placement for specific operations, e.g:

        .. code-block:: python

            tf.debugging.set_log_device_placement(True)

            # train my_model on GPU:1
            with tf.device("/GPU:1"):
                ... # train code goes here.

        then make sure you correctly create your model during inference to avoid any potential errors.

        .. code-block:: python

            # my_model_gpu is a trained on GPU:0, with weight and tokenizer to file
            with tf.device("/GPU:0"):
                my_inference_model = build_model() # build_model
                my_inference_model.set_weights(my_model_gpu.get_weights())
                ... # inference code goes here.

.. note::
    Tensorflow provides ``/GPU:{device_id}`` where ``device_id`` is our GPU/CPU ids. This is useful if you have a multiple CPUs/GPUs setup.
    For most use-case ``/GPU:0`` will do the job.

    You can get the specific device with

    .. code-block:: python

        tf.config.list_physical_devices("GPU") # or CPU

.. _tensorflow-impl:

Tensorflow Implementation
"""""""""""""""""""""""""

.. note::
    refers to `Tensorflow gallery <https://github.com/bentoml/gallery/blob/master/tensorflow/sentiment-analysis-gpu/sentiment-analysis-gpu.ipynb>`_ for the complete version.

.. code-block:: python

    # bento_svc.py
    import bentoml
    from bentoml.adapters import JsonInput
    from bentoml.frameworks.keras import KerasModelArtifact
    from bentoml.service.artifacts.common import PickleArtifact

    @bentoml.env(pip_packages=['tensorflow', 'scikit-learn', 'pandas'] ,\
          docker_base_image="bentoml/model-server:0.12.1-py38-gpu")
    @bentoml.artifacts([KerasModelArtifact('model'), PickleArtifact('tokenizer')])
    class TensorflowService(bentoml.BentoService):

        @api(input=JsonInput())
        def predict(self, parsed_json):
            return self.artifacts.model.predict(input_data)

.. code-block:: python

    # bento_packer.py
    from bento_svc import TensorflowService

    # OPTIONAL: to remove tf memory limit on our card
    config.experimental.set_memory_growth(gpu[0], True)

    model = load_model()
    tokenizer = load_tokenizer()

    bento_svc = TensorflowService()
    bento_svc.pack('model', model)
    bento_svc.pack('tokenizer', tokenizer)

    saved_path = bento_svc.save()


2.4 PyTorch
^^^^^^^^^^^

.. note::
    Since PyTorch bundled CUDNN and NCCL runtime with the python library the *RECOMMENDED* way to run your PyTorch service is to install PyTorch with conda
    via BentoML `@env <http://localhost:8000/api/bentoml.html#env>`_:

    .. code-block:: python

        @env(conda_dependencies=['pytorch', 'torchtext', 'cudatoolkit=11.1'], conda_channels=['pytorch', 'nvidia'],

PyTorch provides a more pythonic way to define device for our deep learning model. This can be used through training and inference tasks

.. code-block:: python

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

.. note::
    PyTorch provides users with **OPTIONAL** ``cuda:{device_id}`` or ``cpu:{device_id}`` to explicitly assign GPU if the vendors contain multiple GPUs or CPUs.
    For mose use-case "cuda" or "cpu" will dynamically allocate GPU resources and fallback to CPU for you.

However, make sure that in our BentoService definition every tensor that is needed for inference *should be cast to the same device as our our model*, see :ref:`pytorch-impl`.

.. note::
    All of the above apply to ``transformers``, ``PytorchLightning`` or any other variant of PyTorch deep learning frameworks.

.. _pytorch-impl:

PyTorch Implementation
""""""""""""""""""""""

.. note::
    refers to `PyTorch gallery <https://github.com/bentoml/gallery/blob/master/pytorch/news-classification-gpu/news-classification.ipynb>`_ for the complete version.

.. code-block:: python

    # bento_svc.py

    from bentoml import BentoService, api, artifacts, env
    from bentoml.adapters import JsonInput, JsonOutput
    from bentoml.frameworks.pytorch import PytorchModelArtifact
    from bentoml.service.artifacts.pickle import PickleArtifact
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @env(conda_dependencies=['pytorch', 'torchtext', 'cudatoolkit=11.1'], conda_channels=['pytorch', 'nvidia'],
     requirements_txt_file=None)
    @artifacts([PytorchModelArtifact("model"), PickleArtifact("tokenizer"), PickleArtifact("vocab")])
    class PytorchService(BentoService):

        def classify_categories(self, sentence):
            text_pipeline, _ = get_pipeline(self.artifacts.tokenizer, self.artifacts.vocab)
            with torch.no_grad():
                # since we want to run our inference tasks with GPU, we need to cast
                # our text and offsets to GPU
                text = torch.tensor(text_pipeline(sentence)).to(device)
                offsets = torch.tensor([0]).to(device)
                output = self.artifacts.model(text, offsets=offsets)
                return output.argmax(1).item() + 1

        @api(input=JsonInput(), output=JsonOutput())
        def predict(self, parsed_json):
            label = self.classify_categories(parsed_json.get("text"))
            return {'categories': self.label[label]}

.. code-block:: python

    # bento_packer.py

    import torch

    from bento_svc import PytorchService

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer, vocab = get_tokenizer_vocab()
    vocab_size, embedding_size, num_class = get_model_params(vocab)

    # here we assign our inference model to the defined device
    model = TextClassificationModel(vocab_size, embedding_size, num_class).to(device)
    model.load_state_dict(torch.load("model/pytorch_model.pt"))
    model.eval()

    bento_svc = PytorchService()

    bento_svc.pack("model", model)
    bento_svc.pack("tokenizer", tokenizer)
    bento_svc.pack("vocab", vocab)
    saved_path = bento_svc.save()

2.5 ONNX
^^^^^^^^

User only need to install ``onnxruntime-gpu`` to be able to run their ONNX model with GPU. It will automatically fallback to CPU if no GPUs are found.

.. note::
    ONNX use-case is dependent on the base deep learning framework user choose to build their model on. This guide will provide
    PyTorch to ONNX use-case. Contributions are welcome for others deep learning frameworks.

User can check if GPU is running for their ``InferenceSession`` with ``get_providers()``:

.. code-block:: python

    cuda = "CUDA" in session.get_providers()[0] # True if you have a GPU

Some notes with regarding to building ONNX services:

- as shown with :ref:`onnx-impl` below, make sure that you setup a correct input and outputs for your ONNX models to avoid any errors.
- your input should be a ``numpy`` array, refers to ``to_numpy()`` for example.

.. _onnx-impl:

ONNX Implementation
"""""""""""""""""""

.. note::
    refers to `ONNX gallery <https://github.com/bentoml/gallery/blob/master/onnx/news-classification-gpu/news-classification-gpu.ipynb>`_ for the complete version.

.. code-block:: python

    # bento_svc.py
    import torch
    from bentoml import BentoService, api, env, artifacts
    from bentoml.adapters import JsonInput, JsonOutput
    from bentoml.frameworks.onnx import OnnxModelArtifact
    from bentoml.service.artifacts.pickle import PickleArtifact
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def to_numpy(tensor):
        return tensor.detach().cpu().clone().numpy() if tensor.requires_grad else tensor.cpu().clone().numpy()


    @env(infer_pip_packages=False, pip_packages=['onnxruntime-gpu'])
    @artifacts(
        [OnnxModelArtifact('model', backend='onnxruntime-gpu'), PickleArtifact('tokenizer'), PickleArtifact('vocab')])
    class OnnxService(BentoService):

        def classify_categories(self, sentence):
            text_pipeline, _ = get_pipeline(self.artifacts.tokenizer, self.artifacts.vocab)
            text = to_numpy(torch.tensor(text_pipeline(sentence)).to(device))
            tensor_name = self.artifacts.model.get_inputs()[0].name
            output_name = self.artifacts.model.get_outputs()[0].name
            onnx_inputs = {tensor_name: text}

            try:
                r = self.artifacts.model.run([output_name], onnx_inputs)[0]
                return r.argmax(1).item() + 1
            except (RuntimeError, InvalidArgument) as e:
                print(f"ERROR with shape: {onnx_inputs[tensor_name].shape} - {e}")

        @api(input=JsonInput(), output=JsonOutput())
        def predict(self, parsed_json):
            sentence = parsed_json.get('text')
            return {'categories': self.label[self.classify_categories(sentence)]}

.. code-block:: python

    import torch
    from bento_svc import OnnxService

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, vocab = get_tokenizer_vocab()
    vocab_size, embedding_size, num_class = get_model_params(vocab)
    model = TextClassificationModel(vocab_size, embedding_size, num_class).to(device)
    model.load_state_dict(torch.load("model/pytorch_model.pt"))
    model.eval()

    # a dummy input is required for onnx model. User has to make sure to correctly set dimension of this input
    # to match with given model inputs. e.g:
    #
    # an alexnet models will take in a 224x224 images so dummy inputs will have a static shape [3, 224,224].
    #
    # however, our new categorization tasks requires a variable in length of our input variables, thus
    # our dummy input should have a dynamic shape [vocab_size].
    #
    # ONNX also only takes torch.LongTensor or torch.cuda.LongTensor so remember to cast correctly.
    # we can handle dynamic axes (vocab_size in this case) with ``dynamic_axes=`` as shown below.

    inp = torch.rand(vocab_size).long().to(device)

    torch.onnx.export(model, inp, onnx_model_path, export_params=True, opset_version=11, do_constant_folding=True,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "vocab_size"}, "output": {0: "vocab_size"}})

    bento_svc = OnnxService()
    bento_svc.pack("model", onnx_model_path)
    bento_svc.pack("tokenizer", tokenizer)
    bento_svc.pack("vocab", vocab)
    saved_path = bento_svc.save()