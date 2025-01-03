======================
Load and manage models
======================

BentoML offers simple APIs for you to load, store and manage AI models.

Understand the Model Store
--------------------------

BentoML provides a local Model Store to save and manage models, which is essentially a local file directory maintained by BentoML. It is useful in several scenarios including:

- **Private model management**: For private models fine-tuned or trained for specific tasks, using BentoML's Model Store offers a secure and efficient way to store, version, and access them.
- **Model cataloging**: BentoML's Model Store facilitates easy cataloging and versioning of models, enabling you to maintain a clear record of model iterations and switch.

Save a model
------------

You can register a model to the Model Store using ``bentoml.models.create()`` with a context manager to ensure proper cleanup and saving of the model. For example, you can save a Hugging Face Transformers pipeline into the Model Store as below:

.. code-block:: python

    import transformers
    import bentoml

    model= "sshleifer/distilbart-cnn-12-6"
    task = "summarization"
    pipeline = transformers.pipeline(task, model=model)

    with bentoml.models.create(
        name='summarization-model', # Name of the model in the Model Store
    ) as model_ref:
        pipeline.save_pretrained(model_ref.path)
        print(f"Model saved: {model_ref}")

By default, all models downloaded to the Model Store are saved in the directory ``/home/user/bentoml/models/``, with each of them assigned a specific subdirectory. For example, the above code snippet will save the summarization model to ``/home/user/bentoml/models/summarization-model/``. You can retrieve the path of the saved model by using its ``path`` property.

If you have an existing model on disk, you can import it into the BentoML Model Store through ``shutil``.

.. code-block:: python

    import shutil
    import bentoml

    local_model_dir = '/path/to/your/local/model/directory'

    with bentoml.models.create(
        name='my-local-model', # Name of the model in the Model Store
    ) as model_ref:
        # Copy the entire model directory to the BentoML Model Store
        shutil.copytree(local_model_dir, model_ref.path, dirs_exist_ok=True)
        print(f"Model saved: {model_ref}")

Load a model
------------

BentoML provides an efficient mechanism for loading AI models to accelerate model deployment, reducing image build time and cold start time.

.. tab-set::

   .. tab-item:: From Hugging Face

      To load a model from Hugging Face (HF), instantiate a ``HuggingFaceModel`` class from ``bentoml.models`` and specify the model ID as shown on HF. For a gated Hugging Face model, remember to export your `Hugging Face API token <https://huggingface.co/docs/hub/en/security-tokens>`_ as environment variables before loading the model.

      Here is an example:

      .. code-block:: python

         import bentoml
         from bentoml.models import HuggingFaceModel
         from transformers import AutoModelForSequenceClassification, AutoTokenizer

         @bentoml.service(resources={"cpu": "200m", "memory": "512Mi"})
         class MyService:
             # Specify a model from HF with its ID
             model_path = HuggingFaceModel("google-bert/bert-base-uncased")

             def __init__(self):
                 # Load the actual model and tokenizer within the instance context
                 self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                 self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

      By default, ``HuggingFaceModel`` returns the downloaded model path as a string, which means you can directly pass the path into libraries like ``transformers`` for model loading.

      If your model is hosted in a private repository, specify your endpoint URL through the ``endpoint`` parameter, which defaults to ``https://huggingface.co/``.

      .. code-block:: python

         model_path = HuggingFaceModel("your_model_id", endpoint="https://my.huggingface.co/")

      After deploying the HF model to BentoCloud, you can view and verify it on the Bento details page. It is indicated with the HF icon. Clicking it redirects you to the model page on HF.

      .. image:: ../../_static/img/build-with-bentoml/model-loading-and-management/hf-model-on-bentocloud.png
         :alt: Hugging Face model marked with an icon on BentoCloud console

   .. tab-item:: From the Model Store or BentoCloud

      To load a model from the local Model Store or BentoCloud, instantiate a ``BentoModel`` from ``bentoml.models`` and specify its model tag. Make sure the model is stored locally or available in BentoCloud.

      Here is an example:

      .. code-block:: python

         import bentoml
         from bentoml.models import BentoModel
         import joblib

         @bentoml.service(resources={"cpu": "200m", "memory": "512Mi"})
         class MyService:
             # Define model reference at the class level
             # Load a model from the Model Store or BentoCloud
             iris_ref = BentoModel("iris_sklearn:latest")

             def __init__(self):
                 self.iris_model = joblib.load(self.iris_ref.path_of("model.pkl"))

      By default, ``__get__`` from ``BentoModel`` returns a ``bentoml.Model`` object, which requires additional tools like ``joblib.load`` to load the model data.

When using ``HuggingFaceModel`` and ``BentoModel``, you must load the model from the class scope of a Service. Defining the model as a class variable declares it as a dependency of the Service, ensuring the models are referenced by the Bento when transported and deployed. If you call these two APIs within the constructor of a Service class, the model will not be referenced by the Bento. As a result, it will not be pushed or deployed, leading to a model ``NotFound`` error.

.. note::

    BentoML accelerates model loading in two key ways. First, when using ``HuggingFaceModel`` or ``BentoModel``, models are downloaded during image building rather than Service startup. The downloaded models are cached and mounted directly into containers, significantly reducing cold start time and improving scaling performance, especially for large models. Second, BentoML optimizes the actual loading process itself with parallel loading using safetensors. Instead of loading model weights sequentially, multiple parts of the model are loaded simultaneously.

For more information, see :doc:`/reference/bentoml/stores`.

Manage models
-------------

Saving a model to the Model Store and retrieving it are the two most common use cases for managing models. In addition to them, you can also perform other operations by using the BentoML CLI or management APIs.

CLI commands
^^^^^^^^^^^^

You can perform the following operations on models by using the BentoML CLI.

.. tab-set::

    .. tab-item:: List

        To list all available models:

        .. code-block:: bash

            $ bentoml models list

            Tag                                   Module  Size      Creation Time
            summarization-model:btwtmvu5kwqc67i3          1.14 GiB  2023-12-18 03:25:10

    .. tab-item:: Get

        To retrieve the information of a specific model:

        .. code-block:: bash

            $ bentoml models get summarization-model:latest

            name: summarization-model
            version: btwtmvu5kwqc67i3
            module: ''
            labels: {}
            options: {}
            metadata:
            model_name: sshleifer/distilbart-cnn-12-6
            task_name: summarization
            context:
            framework_name: ''
            framework_versions: {}
            bentoml_version: 1.1.10.post84+ge2e9ccc1
            python_version: 3.9.16
            signatures: {}
            api_version: v1
            creation_time: '2023-12-18T03:25:10.972481+00:00'

    .. tab-item:: Import/Export

        You can export a model in the BentoML Model Store as a standalone archive file and share it between teams or move it between different build stages. For example:

        .. code-block:: bash

            $ bentoml models export summarization-model:latest .

            Model(tag="summarization-model:btwtmvu5kwqc67i3") exported to ./summarization-model-btwtmvu5kwqc67i3.bentomodel

        .. code-block:: bash

            $ bentoml models import ./summarization-model-btwtmvu5kwqc67i3.bentomodel

            Model(tag="summarization-model:btwtmvu5kwqc67i3") imported

        You can export models to and import models from external storage devices, such as AWS S3, GCS, FTP and Dropbox. For example:

        .. code-block:: bash

            pip install fs-s3fs  *# Additional dependency required for working with s3*
            bentoml models export summarization-model:latest s3://my_bucket/my_prefix/

    .. tab-item:: Pull/Push

        `BentoCloud <https://cloud.bentoml.com/>`_ provides a centralized model repository with flexible APIs and a web console for managing all models created by your team. After you :doc:`log in to BentoCloud </scale-with-bentocloud/manage-api-tokens>`, use ``bentoml models push`` and ``bentoml models pull`` to upload your models to and download them from BentoCloud:

        .. code-block:: bash

            $ bentoml models push summarization-model:latest

            Successfully pushed model "summarization-model:btwtmvu5kwqc67i3"                                                                                                                                                                                           │

        .. code-block:: bash

            $ bentoml models pull summarization-model:latest

            Successfully pulled model "summarization-model:btwtmvu5kwqc67i3"

    .. tab-item:: Delete

        .. code-block:: bash

            $ bentoml models delete summarization-model:latest -y

            INFO [cli] Model(tag="summarization-model:btwtmvu5kwqc67i3") deleted

.. tip::

    Learn more about CLI usage by running ``bentoml models --help``.

Python APIs
^^^^^^^^^^^

In addition to the CLI commands, BentoML also provides equivalent Python APIs for managing models.

.. tab-set::

    .. tab-item:: List

        ``bentoml.models.list`` returns a list of ``bentoml.Model`` instances:

        .. code-block:: python

            import bentoml
            models = bentoml.models.list()

    .. tab-item:: Import/Export

        .. code-block:: python

            import bentoml
            bentoml.models.export_model('iris_clf:latest', '/path/to/folder/my_model.bentomodel')

        .. code-block:: python

            bentoml.models.import_model('/path/to/folder/my_model.bentomodel')

        You can export models to and import models from external storage devices, such as AWS S3, GCS, FTP and Dropbox. For example:

        .. code-block:: python

            bentoml.models.import_model('s3://my_bucket/folder/my_model.bentomodel')

    .. tab-item:: Push/Pull

        If you :doc:`have access to BentoCloud </scale-with-bentocloud/manage-api-tokens>`, you can also push local models to or pull models from it.

        .. code-block:: python

            import bentoml
            bentoml.models.push("summarization-model:latest")

        .. code-block:: python

            bentoml.models.pull("summarization-model:latest")

    .. tab-item:: Delete

        .. code-block:: python

            import bentoml
            bentoml.models.delete("summarization-model:latest")
