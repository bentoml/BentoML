============
TensorRT-LLM
============

`TensorRT-LLM <https://github.com/NVIDIA/TensorRT-LLM>`_ provides a streamlined Python API to efficiently define large language models (LLMs) and build optimized `TensorRT <https://developer.nvidia.com/tensorrt>`_ engines. These engines leverage state-of-the-art optimizations for high-performance inference on NVIDIA GPUs.

This document demonstrates how to build an LLM application using BentoML and TensorRT-LLM.

All the source code in this tutorial is available in `the BentoTRTLLM repository <https://github.com/bentoml/BentoTRTLLM>`_.

Prerequisites
-------------

- Python 3.10+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read :doc:`/get-started/quickstart` first.
- You have installed `Docker <https://docs.docker.com/engine/install/>`_, which will be used to create a container environment to run TensorRT-LLM.
- If you want to test the project locally, you need a Nvidia GPU with at least 20G VRAM.
- (Optional) We recommend you create a virtual environment for dependency isolation. See the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or the `Python documentation <https://docs.python.org/3/library/venv.html>`_ for details.

Set up the environment
----------------------

Clone the project repository and TensorRT-LLM repository.

.. code-block:: bash

    git clone https://github.com/bentoml/BentoTRTLLM.git
    cd BentoTRTLLM/llama-3-8b-instruct
    git clone -b v0.10.0 https://github.com/NVIDIA/TensorRT-LLM.git
    cd TensorRT-LLM

Create the base Docker environment to compile the model. This tutorial uses ``Meta-Llama-3-8B-Instruct``, which requires you to `accept relevant conditions to gain access <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`_. You can choose any other model supported by TensorRT-LLM based on your needs.

.. code-block:: bash

    git lfs install
    git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
    docker run --rm --runtime=nvidia --gpus all --volume ${PWD}:/TensorRT-LLM --entrypoint /bin/bash -it --workdir /TensorRT-LLM nvidia/cuda:12.1.0-devel-ubuntu22.04

Install dependencies inside the Docker container. Note that TensorRT-LLM requires Python 3.10.

.. code-block:: bash

    apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev

    # Install the stable version (corresponding to the cloned branch) of TensorRT-LLM.
    pip3 install tensorrt_llm==0.10.0 -U --extra-index-url https://pypi.nvidia.com
    pip3 install --force-reinstall -U --extra-index-url https://pypi.nvidia.com tensorrt-cu12==10.0.1

    # Log in to huggingface-cli
    # You can get your token from huggingface.co/settings/token
    apt-get install -y git
    huggingface-cli login --token *****

Build the Llama 3 8B model using a single GPU and BF16.

.. code-block:: bash

    python3 examples/llama/convert_checkpoint.py --model_dir ./Meta-Llama-3-8B-Instruct \
                --output_dir ./tllm_checkpoint_1gpu_bf16 \
                --dtype bfloat16

    trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_bf16 \
                --output_dir ./tmp/llama/8B/trt_engines/bf16/1-gpu \
                --gpt_attention_plugin bfloat16 \
                --gemm_plugin bfloat16 \
                --max_batch_size 2048 \
                --max_input_len 2048 \
                --max_num_tokens 2048 \
                --multiple_profiles enable \
                --paged_kv_cache enable \
                --use_paged_context_fmha enable

The model should be successfully built now. Exit the Docker image.

.. code-block:: bash

    exit

Clone the ``tensorrtllm_backend`` repository.

.. code-block:: bash

    cd ..
    git clone -b v0.10.0 https://github.com/triton-inference-server/tensorrtllm_backend.git

The ``BentoTRTLLM/`` directory should have one ``TenosrRT-LLM/`` directory and one ``tensorrtllm_backend/`` directory. Copy the model.

.. code-block:: bash

    cd tensorrtllm_backend
    cp ../TensorRT-LLM/tmp/llama/8B/trt_engines/bf16/1-gpu/* all_models/inflight_batcher_llm/tensorrt_llm/1/

Set the ``tokenizer_dir`` and ``engine_dir`` paths as well as model configurations.

.. code-block:: bash

    HF_LLAMA_MODEL=TensorRT-LLM/Meta-Llama-3-8B-Instruct
    ENGINE_PATH=tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/1

    python3 tools/fill_template.py -i all_models/inflight_batcher_llm/preprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:2048,preprocessing_instance_count:1

    python3 tools/fill_template.py -i all_models/inflight_batcher_llm/postprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},tokenizer_type:auto,triton_max_batch_size:2048,postprocessing_instance_count:8

    python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:2048,decoupled_mode:True,bls_instance_count:1,accumulate_tokens:False

    python3 tools/fill_template.py -i all_models/inflight_batcher_llm/ensemble/config.pbtxt triton_max_batch_size:2048

    python3 tools/fill_template.py -i all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:2048,decoupled_mode:True,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.9,exclude_input_in_output:True,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:True

For more information on configuring the TensorRT-LLM backend, see the `its official documentation <https://github.com/triton-inference-server/tensorrtllm_backend/>`_.

Import the model
----------------

With the model compiled, import the model to the BentoML :doc:`/guides/model-store`, a local directory to store and manage models. You can retrieve this model later in other services.

Make sure you are in the ``llama-3-8b-instruct/`` directory and run `the pack_model.py <https://github.com/bentoml/BentoTRTLLM/blob/main/llama-3-8b-instruct/pack_model.py>`_ script to import the model.

.. code-block:: bash

    pip install bentoml
    python pack_model.py

To verify the result, run:

.. code-block:: bash

    $ bentoml models list

    Tag                                                                           Size       Creation Time
    meta-llama--meta-llama-3-8b-instruct-trtllm-rtx4000:k72ks4cofcjrsw62          45.77 GiB  2024-07-30 04:01:05

Create a BentoML Service
------------------------

With the model imported, the next step is to create a :doc:`BentoML Service </guides/services>`, which can serve the model with custom logic and expose API endpoints for interaction.

First, create a Docker container environment for TensorRT-LLM. The following command runs a Docker container with GPU support, mapping the local project directory and BentoML home directory into the container.

.. code-block:: bash

    docker run --runtime=nvidia --gpus all -v ${PWD}:/BentoTRTLLM -v ~/bentoml:/root/bentoml -p 3000:3000 --entrypoint /bin/bash -it --workdir /BentoTRTLLM nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3

Next, create a ``service.py`` file (already available in the `repo cloned <https://github.com/bentoml/BentoTRTLLM/blob/main/llama-3-8b-instruct/service.py>`_) for defining the serving logic of the model. Here is a breakdown of the key code snippets in the file.

1. Use constants for controlling output length and prompt templates as guidelines for the model to answer queries. They will be referenced in the code implementations later.

   .. code-block:: python
        :caption: `service.py`

        # Maximum number of tokens the model can generate
        MAX_TOKENS = 1024

        # Default system prompt defining the AI assistant's behavior
        SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        # Template for formatting user and system prompts
        PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """
        ...

2. Create a class ``TRTLLM`` to define the required methods and use the ``@bentoml.service`` decorator to mark it as a BentoML Service. You can optionally set runtime :doc:`configurations </guides/configurations>` like ``timeout``.

   .. code-block:: python
        :caption: `service.py`

        ...
        # BentoML Service definition with specific configurations
        @bentoml.service(
            name="bentotrtllm-llama3-8b-insruct-service",
            traffic={
                "timeout": 300,
            },
            resources={
                "gpu": 1, # The number of GPUs used
                "gpu_type": "nvidia-a100-80gb", # The resource type on BentoCloud
            },
        )
        class TRTLLM:

            # Retrieve the model from the BentoML Model Store
            bento_model_ref = bentoml.models.get(BENTO_MODEL_TAG)
        ...

3. Within the class, the following methods initialize and manage the inference pipeline, enabling the Service to configure, communicate, and control model inference operations through the Triton inference server.

   .. code-block:: python
        :caption: `service.py`

        ...
        class TRTLLM:

            # Retrieve the model from the BentoML Model Store
            bento_model_ref = bentoml.models.get(BENTO_MODEL_TAG)
            def __init__(self) -> None:

                # Get the path to the model
                target_dir = self.bento_model_ref.path
                # Command to launch the Triton server with a script
                cmd = ["python3", "tensorrtllm_backend/scripts/launch_triton_server.py"]
                flags = [
                    "--model_repo",
                    "tensorrtllm_backend/all_models/inflight_batcher_llm",
                    "--world_size",
                    "1",
                ]
                # Launch the Triton server as a subprocess
                self.launcher = subprocess.Popen(
                    cmd + flags,
                    env={**os.environ},
                    cwd=target_dir,
                )
                # Initialize a placeholder for a gRPC client, which will later be used to communicate with the Triton server
                self._grpc_client = None

            def start_grpc_stream(self) -> grpcclient.InferenceServerClient:
                # Create and return a gRPC client if it doesn't exist
                if self._grpc_client:
                    return self._grpc_client

                self._grpc_client = grpcclient.InferenceServerClient(
                    url=f"localhost:8001", verbose=False
                )
                return self._grpc_client

            def prepare_tensor(self, name, input):
                # Convert NumPy to Triton-compatible input tensor
                from tritonclient.utils import np_to_triton_dtype

                t = grpcclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
                t.set_data_from_numpy(input)
                return t

            def create_request(
                self,
                prompt,
                streaming,
                request_id,
                output_len,
                temperature=1.0,
            ):
                # Prepare input data for the model
                input0 = [[prompt]]
                input0_data = np.array(input0).astype(object)
                output0_len = np.ones_like(input0).astype(np.int32) * output_len
                streaming_data = np.array([[streaming]], dtype=bool)
                temperature_data = np.array([[temperature]], dtype=np.float32)

                # Create input tensors
                inputs = [
                    self.prepare_tensor("text_input", input0_data),
                    self.prepare_tensor("max_tokens", output0_len),
                    self.prepare_tensor("stream", streaming_data),
                    self.prepare_tensor("temperature", temperature_data),
                ]

                # Specify requested outputs
                outputs = []
                outputs.append(grpcclient.InferRequestedOutput("text_output"))

                # Issue the asynchronous sequence inference
                return {
                    "model_name": "ensemble",
                    "inputs": inputs,
                    "outputs": outputs,
                    "request_id": str(request_id),
                }
        ...

4. Define an asynchronous API endpoint to handle requests. The ``generate`` method formats the prompt using a template, initializes a gRPC client, and generates streamed responses.

   .. code-block:: python
        :caption: `service.py`

        ...
            # Define an API endpoint to generate responses
            @bentoml.api
            async def generate(
                self,
                prompt: str = "Explain superconductors in plain English",
                system_prompt: Optional[str] = SYSTEM_PROMPT,
                max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
            ) -> AsyncGenerator[str, None]:
            
                # Format the prompt using the template
                if system_prompt is None:
                    system_prompt = SYSTEM_PROMPT
                    
                # Format the prompt using the predefined template
                prompt = PROMPT_TEMPLATE.format(user_prompt=prompt, system_prompt=system_prompt)
                        
                # Get or create a gRPC client
                grpc_client_instance = self.start_grpc_stream()
                
                # Define an asynchronous generator that constructs and yields inference requests
                async def input_generator():
                    yield self.create_request(
                        prompt,
                        streaming=True, 
                        request_id=random.randint(1, 9999999),
                        output_len=max_tokens,
                    )

                # Start streaming inference by sending generated requests to the Triton server
                response_iterator = grpc_client_instance.stream_infer(
                    inputs_iterator=input_generator(),
                )

                # Asynchronously iterate over responses from the server
                try:
                    async for response in response_iterator:
                        result, error = response
                        if result:
                            result = result.as_numpy("text_output")
                            yield result[0].decode("utf-8")
                        else:
                            yield json.dumps({"status": "error", "message": error.message()})

                except grpcclient.InferenceServerException as e:
                    # Handle exceptions from the Triton server, logging them for debugging
                    print(f"InferenceServerException: {e}")

With the ``service.py`` file ready, install the dependencies within the container.

.. code-block:: bash

    pip install -r requirements.txt

Run ``bentoml serve`` in the project directory to start the Service.

.. code-block:: bash

    $ bentoml serve .

    2024-07-30T04:10:56+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:TRTLLM" listening on http://localhost:3000 (Press CTRL+C to quit)
    I0730 04:10:58.017996 358 pinned_memory_manager.cc:277] "Pinned memory pool is created at '0x7f1eaa000000' with size 268435456"
    I0730 04:10:58.018423 358 cuda_memory_manager.cc:107] "CUDA memory pool is created on device 0 with size 67108864"
    ...

The server is active at `http://localhost:3000 <http://localhost:3000>`_. You can interact with it in different ways.

.. tab-set::

    .. tab-item:: CURL

        .. code-block:: bash

            curl -X 'POST' \
                'http://localhost:3000/generate' \
                -H 'accept: text/event-stream' \
                -H 'Content-Type: application/json' \
                -d '{
                "prompt": "Explain superconductors in plain English",
                "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don'\''t know the answer to a question, please don'\''t share false information.",
                "max_tokens": 1024
            }'

    .. tab-item:: Python client

        .. code-block:: python

            import bentoml

            with bentoml.SyncHTTPClient("http://localhost:3000") as client:
                response_generator = client.generate(
                    prompt="Explain superconductors like I'm five years old",
                    max_tokens=1024
                )
                for response in response_generator:
                    print(response, end='')

    .. tab-item:: Swagger UI

        Visit `http://localhost:3000 <http://localhost:3000/>`_, scroll down to **Service APIs**, and click **Try it out**. In the **Request body** box, enter your prompt and click **Execute**.

        .. image:: ../../_static/img/use-cases/large-language-models/tensorrt-llm/service-ui.png

Deploy to BentoCloud
--------------------

After the Service is ready, you can deploy the project to BentoCloud for better management and scalability. `Sign up <https://www.bentoml.com/>`_ for a BentoCloud account and get $10 in free credits.

First, specify a configuration YAML file (``bentofile.yaml``) to define the build options for a :doc:`Bento </guides/build-options>`, the unified distribution format in BentoML containing source code, Python packages, model references, and so on. Here is an example file:

.. code-block:: yaml
    :caption: `bentofile.yaml`

    service: "service:TRTLLM"
    labels:
      owner: bentoml-team
      stage: demo
    include:
      - "service.py"
      - "pack_model.py"
    python:
      requirements_txt: "./requirements.txt"
      lock_packages: false
    docker:
      base_image: "nvcr.io/nvidia/tritonserver:24.03-trtllm-python-py3"

:ref:`Log in to BentoCloud <bentocloud/how-tos/manage-access-token:Log in to BentoCloud using the BentoML CLI>` by running ``bentoml cloud login``, then run the following command to deploy the Service.

.. code-block:: bash

    bentoml deploy .

Once the Deployment is up and running on BentoCloud, you can access it via the exposed URL.

.. note::

   For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image</guides/containerization>`.
