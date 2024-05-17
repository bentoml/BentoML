===========
Model Store
===========

BentoML provides a local Model Store to save and manage models, which is essentially a local file directory maintained by BentoML. This document explains how to use the BentoML Model Store.

When should you use the Model Store?
------------------------------------

While it's straightforward to download and use pre-trained models from public model hubs like Hugging Face directly within a ``service.py`` file for simple use cases, more complex scenarios often require a more organized approach to model management. We recommend you use the BentoML Model Store in the following scenarios:

- **Private model management**: If you are working with private models that have been fine-tuned or trained from scratch for specific tasks, using BentoML's Model Store offers a secure and efficient way to store, version, and access these models across your projects.
- **Model cataloging**: BentoML's Model Store facilitates easy cataloging and versioning of models, enabling you to maintain a clear record of model iterations and switch between different model versions as required.
- **Model downloading acceleration in BentoCloud**: For deployment on BentoCloud, the Model Store improves the cold start time of model downloading. BentoCloud caches models to expedite their availability and supports streaming loading of models directly to GPU memory.

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

Retrieve a model
----------------

To retrieve a model from the BentoML Model Store, use the ``get`` method.

.. code-block:: python

    import bentoml
    bento_model: bentoml.Model = bentoml.models.get("summarization-model:latest")

    # Print related attributes of the model object.
    print(bento_model.tag)
    print(bento_model.path)

``bentoml.models.get`` returns a ``bentoml.Model`` instance, linking to a saved model entry in the BentoML Model Store. You can then use the instance to get model information like tag, labels, and file system paths, or create a :doc:`Service </guides/services>` on top of it.

For example, you can load the model into a Transformers pipeline from the ``path`` provided by the ``bentoml.Model`` instance as below, see more in :doc:`/get-started/quickstart`.

.. code-block:: python

    import bentoml
    from transformers import pipeline

    @bentoml.service
    class Summarization:
        # Define the model as a class variable
        model_ref = bentoml.models.get("summarization-model")

        def __init__(self) -> None:
            # Load model into pipeline
            self.pipeline = pipeline('summarization', self.model_ref.path)

        @bentoml.api
        def summarize(self, text: str = EXAMPLE_INPUT) -> str:
            ...


Models must be retrieved from the class scope of a Service. Defining the model as a class variable declares it as a dependency of the Service, ensuring the models are referenced by the Bento when transported and deployed.

.. warning::

    If ``bentoml.models.get()`` is called inside the constructor of a Service class, the model will not be referenced by the Bento therefore not pushed or deployed, leading to model not found issues.


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

        `BentoCloud <https://cloud.bentoml.com/>`_ provides a centralized model repository with flexible APIs and a web console for managing all models created by your team. After you :doc:`log in to BentoCloud </bentocloud/how-tos/manage-access-token>`, use ``bentoml models push`` and ``bentoml models pull`` to upload your models to and download them from BentoCloud:

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

        If you :doc:`have access to BentoCloud </bentocloud/how-tos/manage-access-token>`, you can also push local models to or pull models from it.

        .. code-block:: python

            import bentoml
            bentoml.models.push("summarization-model:latest")

        .. code-block:: python

            bentoml.models.pull("summarization-model:latest")

    .. tab-item:: Delete

        .. code-block:: python

            import bentoml
            bentoml.models.delete("summarization-model:latest")
