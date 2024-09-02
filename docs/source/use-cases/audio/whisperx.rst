============================
WhisperX: Speech recognition
============================

Speech recognition involves the translation of spoken words into text. It is widely used in AI scenarios like virtual assistants, voice-controlled devices, and automated transcription services.

This document demonstrates how to create a speech recognition application with BentoML. It is inspired by the `WhisperX <https://github.com/m-bain/whisperX>`_ project.

All the source code in this tutorial is available in the `BentoWhisperX GitHub repository <https://github.com/bentoml/BentoWhisperX>`_.

Prerequisites
-------------

- Python 3.8+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read :doc:`/get-started/quickstart` first.
- If you want to test this project locally, install FFmpeg on your system.
- Gain access to the model used in this project: `pyannote/segmentation-3.0 <https://huggingface.co/pyannote/segmentation-3.0>`_.
- (Optional) We recommend you create a virtual environment for dependency isolation. See the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or the `Python documentation <https://docs.python.org/3/library/venv.html>`_ for details.

Install dependencies
--------------------

Clone the project repository and install all the dependencies.

.. code-block:: bash

    git clone https://github.com/bentoml/BentoWhisperX.git
    cd BentoWhisperX
    pip install -r requirements.txt

Create a BentoML Service
------------------------

Create a :doc:`BentoML Service </guides/services>` to define the serving logic of this project. Here is an example file in the project:

.. code-block:: python
    :caption: `service.py`

    import bentoml
    import os
    import typing as t

    from pathlib import Path

    LANGUAGE_CODE = "en"


    @bentoml.service(
        traffic={
            "timeout": 30,
            "concurrency": 1,
        },
        resources={
            "gpu": 1,
            "gpu_type": "nvidia_tesla_t4",
        },
    )
    class WhisperX:
        """
        This class is inspired by the implementation shown in the whisperX project.
        Source: https://github.com/m-bain/whisperX
        """

        def __init__(self):
            import torch
            import whisperx

            self.batch_size = 16 # reduce if low on GPU mem
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "int8"
            self.model = whisperx.load_model("large-v2", self.device, compute_type=compute_type, language=LANGUAGE_CODE)
            self.model_a, self.metadata = whisperx.load_align_model(language_code=LANGUAGE_CODE, device=self.device)

        @bentoml.api
        def transcribe(self, audio_file: Path) -> t.Dict:
            import whisperx

            audio = whisperx.load_audio(audio_file)
            result = self.model.transcribe(audio, batch_size=self.batch_size)
            result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, self.device, return_char_alignments=False)

            return result

A breakdown of the Service code:

* The ``@bentoml.service`` decorator is used to define the ``WhisperX`` class as a BentoML Service, specifying additional configurations like timeout and resource allocations (GPU and memory).
* During initialization, this Service does the following:

  - Loads the Whisper model with a specific language code, device, and compute type. It runs on either a GPU or CPU based on availability.
  - Loads an alignment model and metadata for the specified language.

* The Service exposes a ``transcribe`` API endpoint: Takes an audio file path as input, uses the Whisper model to transcribe the audio, and aligns the transcription with the audio using the alignment model and metadata. The transcription result is returned as a dictionary.

Run ``bentoml serve`` to start the Service.

.. code-block:: bash

    $ bentoml serve service:WhisperX

    2024-01-22T02:29:10+0000 [WARNING] [cli] Converting 'WhisperX' to lowercase: 'whisperx'.
    2024-01-22T02:29:11+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:BentoWhisperX" listening on http://localhost:3000 (Press CTRL+C to quit)

The server is active at `http://localhost:3000 <http://localhost:3000>`_. You can interact with it in different ways.

.. tab-set::

    .. tab-item:: CURL

        .. code-block:: bash

            curl -X 'POST' \
                'http://localhost:3000/transcribe' \
                -H 'accept: application/json' \
                -H 'Content-Type: multipart/form-data' \
                -F 'audio_file=@female.wav;type=audio/wav'

    .. tab-item:: Python client

        You can either include an URL or a local path to your audio file in the BentoML :doc:`client </guides/clients>`.

        .. code-block:: python

            from pathlib import Path
            import bentoml

            with bentoml.SyncHTTPClient('http://localhost:3000') as client:
                audio_url = 'https://example.org/female.wav'
                response = client.transcribe(audio_file=audio_url)
                print(response)

    .. tab-item:: Swagger UI

        Visit `http://localhost:3000 <http://localhost:3000/>`_, scroll down to **Service APIs**, and select an audio file for interaction.

        .. image:: ../../_static/img/use-cases/audio/whisperx/service-ui.png

Expected output:

.. code-block:: bash

    {"segments":[{"start":0.009,"end":2.813,"text":" The Hispaniola was rolling scuppers under in the ocean swell.","words":[{"word":"The","start":0.009,"end":0.069,"score":0.0},{"word":"Hispaniola","start":0.109,"end":0.81,"score":0.917},{"word":"was","start":0.83,"end":0.95,"score":0.501},{"word":"rolling","start":0.99,"end":1.251,"score":0.839},{"word":"scuppers","start":1.311,"end":1.671,"score":0.947},{"word":"under","start":1.751,"end":1.932,"score":0.939},{"word":"in","start":1.952,"end":2.012,"score":0.746},{"word":"the","start":2.032,"end":2.132,"score":0.667},{"word":"ocean","start":2.212,"end":2.472,"score":0.783},{"word":"swell.","start":2.512,"end":2.813,"score":0.865}]},{"start":3.494,"end":10.263,"text":"The booms were tearing at the blocks, the rudder was banging to and fro, and the whole ship creaking, groaning, and jumping like a manufactory.","words":[{"word":"The","start":3.494,"end":3.594,"score":0.752},{"word":"booms","start":3.614,"end":3.914,"score":0.867},{"word":"were","start":3.934,"end":4.054,"score":0.778},{"word":"tearing","start":4.074,"end":4.315,"score":0.808},{"word":"at","start":4.335,"end":4.395,"score":0.748},{"word":"the","start":4.415,"end":4.475,"score":0.993},{"word":"blocks,","start":4.495,"end":4.855,"score":0.918},{"word":"the","start":5.236,"end":5.316,"score":0.859},{"word":"rudder","start":5.356,"end":5.576,"score":0.894},{"word":"was","start":5.596,"end":5.717,"score":0.711},{"word":"banging","start":5.757,"end":6.117,"score":0.767},{"word":"to","start":6.177,"end":6.317,"score":0.781},{"word":"and","start":6.377,"end":6.458,"score":0.833},{"word":"fro,","start":6.498,"end":6.758,"score":0.657},{"word":"and","start":7.058,"end":7.159,"score":0.759},{"word":"the","start":7.179,"end":7.259,"score":0.833},{"word":"whole","start":7.299,"end":7.479,"score":0.807},{"word":"ship","start":7.539,"end":7.759,"score":0.79},{"word":"creaking,","start":7.859,"end":8.26,"score":0.774},{"word":"groaning,","start":8.44,"end":8.821,"score":0.75},{"word":"and","start":8.861,"end":8.941,"score":0.837},{"word":"jumping","start":8.981,"end":9.321,"score":0.859},{"word":"like","start":9.382,"end":9.502,"score":0.876},{"word":"a","start":9.542,"end":9.582,"score":0.5},{"word":"manufactory.","start":9.622,"end":10.263,"score":0.886}]}],"word_segments":[{"word":"The","start":0.009,"end":0.069,"score":0.0},{"word":"Hispaniola","start":0.109,"end":0.81,"score":0.917},{"word":"was","start":0.83,"end":0.95,"score":0.501},{"word":"rolling","start":0.99,"end":1.251,"score":0.839},{"word":"scuppers","start":1.311,"end":1.671,"score":0.947},{"word":"under","start":1.751,"end":1.932,"score":0.939},{"word":"in","start":1.952,"end":2.012,"score":0.746},{"word":"the","start":2.032,"end":2.132,"score":0.667},{"word":"ocean","start":2.212,"end":2.472,"score":0.783},{"word":"swell.","start":2.512,"end":2.813,"score":0.865},{"word":"The","start":3.494,"end":3.594,"score":0.752},{"word":"booms","start":3.614,"end":3.914,"score":0.867},{"word":"were","start":3.934,"end":4.054,"score":0.778},{"word":"tearing","start":4.074,"end":4.315,"score":0.808},{"word":"at","start":4.335,"end":4.395,"score":0.748},{"word":"the","start":4.415,"end":4.475,"score":0.993},{"word":"blocks,","start":4.495,"end":4.855,"score":0.918},{"word":"the","start":5.236,"end":5.316,"score":0.859},{"word":"rudder","start":5.356,"end":5.576,"score":0.894},{"word":"was","start":5.596,"end":5.717,"score":0.711},{"word":"banging","start":5.757,"end":6.117,"score":0.767},{"word":"to","start":6.177,"end":6.317,"score":0.781},{"word":"and","start":6.377,"end":6.458,"score":0.833},{"word":"fro,","start":6.498,"end":6.758,"score":0.657},{"word":"and","start":7.058,"end":7.159,"score":0.759},{"word":"the","start":7.179,"end":7.259,"score":0.833},{"word":"whole","start":7.299,"end":7.479,"score":0.807},{"word":"ship","start":7.539,"end":7.759,"score":0.79},{"word":"creaking,","start":7.859,"end":8.26,"score":0.774},{"word":"groaning,","start":8.44,"end":8.821,"score":0.75},{"word":"and","start":8.861,"end":8.941,"score":0.837},{"word":"jumping","start":8.981,"end":9.321,"score":0.859},{"word":"like","start":9.382,"end":9.502,"score":0.876},{"word":"a","start":9.542,"end":9.582,"score":0.5},{"word":"manufactory.","start":9.622,"end":10.263,"score":0.886}]}%

Deploy to BentoCloud
--------------------

After the Service is ready, you can deploy the project to BentoCloud for better management and scalability. `Sign up <https://www.bentoml.com/>`_ for a BentoCloud account and get $10 in free credits.

First, specify a configuration YAML file (``bentofile.yaml``) to define the build options for your application. It is used for packaging your application into a Bento. Here is an example file in the project directory:

.. code-block:: yaml
    :caption: `bentofile.yaml`

    service: "service:WhisperX"
    labels:
      owner: bentoml-team
      project: gallery
    include:
      - "*.py"
    python:
      requirements_txt: "./requirements.txt"
    docker:
      system_packages:
        - ffmpeg
        - git

:ref:`Log in to BentoCloud <bentocloud/how-tos/manage-access-token:Log in to BentoCloud using the BentoML CLI>` by running ``bentoml cloud login``, then run the following command to deploy the project.

.. code-block:: bash

    bentoml deploy .

Once the Deployment is up and running on BentoCloud, you can access it via the exposed URL.

.. image:: ../../_static/img/use-cases/audio/whisperx/whisperx-bentocloud.png

.. note::

   For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image</guides/containerization>`.
