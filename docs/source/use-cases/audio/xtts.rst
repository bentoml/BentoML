====================
XTTS: Text to speech
====================

Text-to-speech machine learning technology can convert written text into spoken words. This may involve analyzing the text, understanding its structure and meaning, and then generating speech that mimics human voice and intonation.

This document demonstrates how to build a text-to-speech application using BentoML, powered by the model `XTTS <https://huggingface.co/coqui/XTTS-v2>`_.

Prerequisites
-------------

- Python 3.9+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read :doc:`/get-started/quickstart` first.
- (Optional) We recommend you create a virtual environment for dependency isolation. See the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or the `Python documentation <https://docs.python.org/3/library/venv.html>`_ for details.

Install dependencies
--------------------

Clone the project repository and install all the dependencies.

.. code-block:: bash

    git clone https://github.com/bentoml/BentoXTTS.git
    cd BentoXTTS
    pip install -r requirements.txt

Create a BentoML Service
------------------------

Define a :doc:`BentoML Service </guides/services>` to customize the serving logic of the model. You can find the following example ``service.py`` file in the cloned repository.

.. code-block:: python
    :caption: `service.py`

    from __future__ import annotations

    import os
    import typing as t
    from pathlib import Path

    import torch
    from TTS.api import TTS

    import bentoml

    MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

    sample_input_data = {
        'text': 'It took me quite a long time to develop a voice and now that I have it I am not going to be silent.',
        'language': 'en',
    }

    @bentoml.service(
        resources={
            "gpu": 1,
            "memory": "8Gi",
        },
        traffic={"timeout": 300},
    )
    class XTTS:
        def __init__(self) -> None:
            self.tts = TTS(MODEL_ID, gpu=torch.cuda.is_available())
        
        @bentoml.api
        def synthesize(
                self,
                context: bentoml.Context,
                text: str = sample_input_data["text"],
                lang: str = sample_input_data["language"],
        ) -> t.Annotated[Path, bentoml.validators.ContentType('audio/*')]:
            output_path = os.path.join(context.temp_dir, "output.wav")
            sample_path = "./female.wav"
            if not os.path.exists(sample_path):
                sample_path = "./src/female.wav"

            self.tts.tts_to_file(
                text,
                file_path=output_path,
                speaker_wav=sample_path,
                language=lang,
                split_sentences=True,
            )
            return Path(output_path)

A breakdown of the Service code:

- ``@bentoml.service`` decorates the class ``XTTS`` to define it as a BentoML Service, configuring resources (GPU and memory) and traffic timeout.
- In the class, the ``__init__`` method initializes an instance of the ``TTS`` model using the ``MODEL_ID`` specified. It checks if a GPU is available and sets the model to use it if so.
- The ``synthesize`` method is defined as an API endpoint. It takes ``context``, ``text``, and ``lang`` as parameters, with defaults provided for ``text`` and ``lang`` in ``sample_input_data``. This method generates an audio file from the provided text and language, using the TTS model. It creates an output file path in the temporary directory (``temp_dir``). A sample WAV file path (``sample_path``) is used for the TTS process.
- The Service calls ``tts.tts_to_file`` to generate the audio file (``output.wav``) based on the provided text and language.

Run ``bentoml serve`` in your project directory to start the Service. Set the environment variable ``COQUI_TTS_AGREED=1`` to agree to the terms of Coqui TTS.

.. code-block:: bash

    $ COQUI_TOS_AGREED=1 bentoml serve .

    2024-01-30T10:06:43+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:XTTS" listening on http://localhost:3000 (Press CTRL+C to quit)

The server is active at `http://localhost:3000 <http://localhost:3000>`_. You can interact with it in different ways.

.. tab-set::

    .. tab-item:: CURL

        .. code-block:: bash

            curl -X 'POST' \
                'http://localhost:3000/synthesize' \
                -H 'accept: */*' \
                -H 'Content-Type: application/json' \
                -d '{
                "text": "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
                "lang": "en"
            }'

    .. tab-item:: BentoML client

        This client returns the audio file as a ``Path`` object. You can use it to access or process the file. See :doc:`/guides/clients` for details.

        .. code-block:: python

            import bentoml

            with bentoml.SyncHTTPClient("http://localhost:3000") as client:
                    result = client.synthesize(
                        text="It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
                        lang="en"
                    )

    .. tab-item:: Swagger UI

        Visit `http://localhost:3000 <http://localhost:3000/>`_, scroll down to **Service APIs**, and click **Try it out**. In the **Request body** box, enter your prompt and click **Execute**.

        .. image:: ../../_static/img/use-cases/audio/xtts/service-ui.png

Deploy to production
--------------------

After the Service is ready, you can deploy the project to BentoCloud for better management and scalability.

First, specify a configuration YAML file (``bentofile.yaml``) to define the build options for your application. It is used for packaging your application into a Bento. Here is an example file in the project:

.. code-block:: yaml
    :caption: `bentofile.yaml`

    service: "service:XTTS"
    labels:
      owner: bentoml-team
      project: gallery
    include:
      - "*.py"
      - "female.wav"
    python:
      requirements_txt: requirements.txt
    envs:
      - name: "COQUI_TOS_AGREED"
        value: 1

Make sure you :doc:`have logged in to BentoCloud </bentocloud/how-tos/manage-access-token>`, then run the following command in your project directory to deploy the application to BentoCloud.

.. code-block:: bash

    bentoml deploy .

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

.. note::

   Alternatively, you can use BentoML to generate an :doc:`OCI-compliant image for a more custom deployment </guides/containerization>`.
