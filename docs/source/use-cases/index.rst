========
Examples
========

This section provides the tutorials for a curated list of example projects to help you learn how BentoML can be used for different scenarios.

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`/use-cases/large-language-models/index`
        :link: /use-cases/large-language-models/index
        :link-type: doc

        Deploy LLM applications with BentoML.

    .. grid-item-card:: :doc:`/use-cases/diffusion-models/index`
        :link: /use-cases/diffusion-models/index
        :link-type: doc

        Deploy diffusion models with BentoML.

    .. grid-item-card:: :doc:`/use-cases/custom-models/index`
        :link: /use-cases/custom-models/index
        :link-type: doc

        Deploy custom models with BentoML.

.. toctree::
    :maxdepth: 1
    :titlesonly:
    :hidden:

    large-language-models/index
    diffusion-models/index
    custom-models/index

See the following lists for a complete collection of BentoML example projects. Browse through different categories to find the example that best suits your needs.

LLMs
----

Serve LLMs with BentoML and vLLM:

- `Llama 3.2 90B <https://github.com/bentoml/BentoVLLM/tree/main/llama3.2-90b-instruct>`_
- `Llama 3.1 70B <https://github.com/bentoml/BentoVLLM/tree/main/llama3.1-70b-instruct-awq>`_
- `Mistral 7B <https://github.com/bentoml/BentoVLLM/tree/main/mistral-7b-instruct>`_
- `Pixtral 12B <https://github.com/bentoml/BentoVLLM/tree/main/pixtral-12b>`_
- `Phi 3 mini <https://github.com/bentoml/BentoVLLM/tree/main/phi-3-mini-4k-instruct>`_
- Check out the `BentoVLLM project <https://github.com/bentoml/BentoVLLM>`_ to see more examples

Build compound AI systems on top of LLMs:

- `Build an AI agent with function calling using BentoML <https://github.com/bentoml/BentoFunctionCalling>`_
- `Build a LangGraph agent application with BentoML <https://github.com/bentoml/BentoLangGraph>`_
- `Build a multi-agent application with CrewAI and BentoML <https://github.com/bentoml/BentoCrewAI>`_
- `Build an AI assistant to ensure LLM safety with BentoML and ShieldGemma <https://github.com/bentoml/BentoShield/>`_
- `Build a RAG system for document ingestion and search with LlamaIndex and BentoML <https://github.com/bentoml/rag-tutorials>`_
- `Build a voice agent with open-source models using Twilio, Pipecat and BentoML <https://github.com/bentoml/BentoVoiceAgent>`_

Image and video
---------------

Serve text-to-image and image-to-image models with BentoML:

- `Stable Diffusion 3.5 Large Turbo <https://github.com/bentoml/BentoDiffusion/tree/main/sd3.5-large-turbo>`_
- `Stable Diffusion 3 Medium <https://github.com/bentoml/BentoDiffusion/tree/main/sd3-medium>`_
- `Stable Diffusion XL Turbo <https://github.com/bentoml/BentoDiffusion/tree/main/sdxl-turbo>`_
- `Stable Video Diffusion <https://github.com/bentoml/BentoDiffusion/tree/main/svd>`_
- `ControlNet <https://github.com/bentoml/BentoDiffusion/tree/main/controlnet>`_
- Check out the `BentoDiffusion project <https://github.com/bentoml/BentoDiffusion>`_ to see more examples

Audio
-----

Serve text-to-speech and speech-to-text models with BentoML:

- `ChatTTS <https://github.com/bentoml/BentoChatTTS>`_
- `XTTS <https://github.com/bentoml/BentoXTTS>`_
- `XTTS with a streaming endpoint <https://github.com/bentoml/BentoXTTSStreaming>`_
- `WhisperX <https://github.com/bentoml/BentoWhisperX>`_
- `Bark <https://github.com/bentoml/BentoBark>`_

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

Custom models
-------------

Serve custom models with BentoML:

- `MLflow <https://github.com/bentoml/BentoML/tree/main/examples/mlflow>`_
- `XGBoost <https://github.com/bentoml/BentoML/tree/main/examples/xgboost>`_
- `scikit-learn <https://github.com/bentoml/BentoML/tree/main/examples/sk-learn>`_

Others
------

- `Build an image captioning inference API with BLIP and BentoML <https://github.com/bentoml/BentoBlip>`_
- `Build a forecasting inference API for time-series data with Moirai and BentoML <https://github.com/bentoml/BentoMoirai/>`_
