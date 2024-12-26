========
Overview
========

This section provides the tutorials for a curated list of example projects to help you learn how BentoML can be used for different scenarios. See the following lists for a complete collection of BentoML example projects. Browse through different categories to find the example that best suits your needs.

LLMs
----

One-command LLM deployment with `OpenLLM <https://github.com/bentoml/OpenLLM>`_:

.. code-block:: bash

   pip install openllm  # or pip3 install openllm
   openllm hello

Deploy an OpenAI-compatible LLM API service with BentoML and vLLM:

- `Llama 3.3 70B <https://github.com/bentoml/BentoVLLM/tree/main/llama3.3-70b-instruct-function-calling>`_
- `Llama 3.2 90B <https://github.com/bentoml/BentoVLLM/tree/main/llama3.2-90b-instruct>`_
- `Llama 3.1 70B <https://github.com/bentoml/BentoVLLM/tree/main/llama3.1-70b-instruct-awq>`_
- `Mistral 7B <https://github.com/bentoml/BentoVLLM/tree/main/mistral-7b-instruct>`_
- `Pixtral 12B <https://github.com/bentoml/BentoVLLM/tree/main/pixtral-12b>`_
- `Phi 3 mini <https://github.com/bentoml/BentoVLLM/tree/main/phi-3-mini-4k-instruct>`_

Customize your LLM inference runtime:

- `vLLM <https://github.com/bentoml/BentoVLLM>`_
- `TensorRT-LLM <https://github.com/bentoml/BentoTRTLLM>`_
- `LMDeploy <https://github.com/bentoml/BentoLMDeploy>`_
- `MLC-LLM <https://github.com/bentoml/BentoMLCLLM>`_
- `SGLang <https://github.com/bentoml/BentoSGLang>`_
- `Hugging Face TGI <https://github.com/bentoml/BentoTGI>`_

Compound AI systems
-------------------

Build and scale compound AI systems with BentoML:

- `Agent: Function calling <https://github.com/bentoml/BentoFunctionCalling>`_
- `Agent: LangGraph <https://github.com/bentoml/BentoLangGraph>`_
- `Multi-agent: CrewAI <https://github.com/bentoml/BentoCrewAI>`_
- `LLM safety: ShieldGemma <https://github.com/bentoml/BentoShield/>`_
- `RAG: LlamaIndex <https://github.com/bentoml/rag-tutorials>`_
- `Phone call agent <https://github.com/bentoml/BentoVoiceAgent>`_
- `Multi-LLM routing <https://github.com/bentoml/llm-router>`_

Image and video
---------------

Serve text-to-image and image-to-image models with BentoML:

- `Stable Diffusion 3.5 Large Turbo <https://github.com/bentoml/BentoDiffusion/tree/main/sd3.5-large-turbo>`_
- `Stable Diffusion 3 Medium <https://github.com/bentoml/BentoDiffusion/tree/main/sd3-medium>`_
- `Stable Diffusion XL Turbo <https://github.com/bentoml/BentoDiffusion/tree/main/sdxl-turbo>`_
- `Stable Video Diffusion <https://github.com/bentoml/BentoDiffusion/tree/main/svd>`_
- `ControlNet <https://github.com/bentoml/BentoDiffusion/tree/main/controlnet>`_
- `ComfyUI workflows as APIs <https://github.com/bentoml/comfy-pack>`_
- Check out the `BentoDiffusion project <https://github.com/bentoml/BentoDiffusion>`_ to see more examples

Audio
-----

Serve text-to-speech and speech-to-text models with BentoML:

- `ChatTTS <https://github.com/bentoml/BentoChatTTS>`_
- `XTTS <https://github.com/bentoml/BentoXTTS>`_
- `XTTS with a streaming endpoint <https://github.com/bentoml/BentoXTTSStreaming>`_
- `WhisperX <https://github.com/bentoml/BentoWhisperX>`_
- `Bark <https://github.com/bentoml/BentoBark>`_
- `Moshi <https://github.com/bentoml/BentoMoshi>`_

Computer vision
---------------

Serve computer vision models with BentoML:

- `YOLO: Object detection <https://github.com/bentoml/BentoYolo>`_
- `ResNet: Image classification <https://github.com/bentoml/BentoResnet>`_

Embeddings
----------

Build embedding inference APIs with BentoML:

- `SentenceTransformers <https://github.com/bentoml/BentoSentenceTransformers>`_
- `CLIP <https://github.com/bentoml/BentoClip>`_
- `ColPali <https://github.com/bentoml/BentoColPali>`_

Custom models
-------------

Serve custom models with BentoML:

- `MLflow <https://github.com/bentoml/BentoMLflow>`_
- `XGBoost <https://github.com/bentoml/BentoXGBoost>`_

Others
------

- `BLIP inference API for image captioning and VQA (Visual Question Answering) <https://github.com/bentoml/BentoBlip>`_
- `Serving Moirai: Time-series forecasting as a service <https://github.com/bentoml/BentoMoirai/>`_
