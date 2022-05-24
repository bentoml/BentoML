================
Preparing Models
================


Save A Trained Model
--------------------

Trained ML models needs to be saved with BentoML API, in order to be served with
BentoML. For most cases, it will be just one line added to your model training pipeline,
invoking a :code:`save_model` call, as demonstrated in the :doc:`tutorial </tutorial>`:

.. code:: python

    bentoml.sklearn.save_model("iris_clf", clf)

    # INFO  [cli] Using default model signature `{"predict": {"batchable": False}}` for sklearn model
    # INFO  [cli] Successfully saved Model(tag="iris_clf:2uo5fkgxj27exuqj", path="~/bentoml/models/iris_clf/2uo5fkgxj27exuqj/")


Optionally, you may attach labels, metadata, or :code:`custom_objects` to be saved
together with your model, e.g.:

.. code:: python

    bentoml.pytorch.save_model(
        "demo_mnist",   # model name in the local model store
        trained_model,  # model instance being saved
        metadata={  # user-defined additional metadata
            "acc": acc,
            "cv_stats": cv_stats,
            "dataset_version": "20210820",
        },
        labels={    # user-defined labels for managing models in Yatai
            "owner": "nlp_team",
            "stage": "dev",
        },
        custom_objects={    # save additional user-defined python objects
            "tokenizer": tokenizer_object,
        }
    )


Retrieve a saved model
----------------------

To load the model instance back into memory, use the framework-specific
:code:`load_model` method. For example:

.. code:: python

    import bentoml
    from sklearn.base import BaseEstimator

    model:BaseEstimator = bentoml.sklearn.load_model("iris_clf:latest")

The :code:`load_model` method is meant for testing and development purpose.


Using Model Runner
------------------



Model Signatures
----------------

.. code:: python

    bentoml.pytorch.save_model(
        "demo_mnist",  # model name in the local model store
        trained_model,  # model instance being saved
        signatures={   # model signatures for running inference
          "predict": {
            "batchable": True,
            "batch_dim": 0,
          }
        }
    )


Batchable Model Runner
----------------------


get, to_runner
testing runner
runner input/output

Model signature
* batchable
* batch_dim

Dynamic batching params
* max_batch_size
* max_latency_ms



Managing Models
---------------

Saved models are stored in BentoML's model store, which is a local file directory
maintained by BentoML. Users can view and manage all saved models via the
:code:`bentoml models` CLI command:

.. tab-set::

    .. tab-item:: Get

       .. code:: bash

          > bentoml models get iris_clf:latest

          name: iris_clf
          version: 2uo5fkgxj27exuqj
          module: bentoml.sklearn
          labels: {}
          options: {}
          metadata: {}
          context:
            framework_name: sklearn
            framework_versions:
              scikit-learn: 1.1.0
            bentoml_version: 1.0.0
            python_version: 3.8.12
          signatures:
            predict:
              batchable: false
          api_version: v1
          creation_time: '2022-05-19T08:36:52.456990+00:00'

    .. tab-item:: List

       .. code:: bash

          > bentoml models list

          Tag                        Module           Size        Creation Time        Path
          iris_clf:2uo5fkgxj27exuqj  bentoml.sklearn  5.81 KiB    2022-05-19 08:36:52  ~/bentoml/models/iris_clf/2uo5fkgxj27exuqj
          iris_clf:nb5vrfgwfgtjruqj  bentoml.sklearn  5.80 KiB    2022-05-17 21:36:27  ~/bentoml/models/iris_clf/nb5vrfgwfgtjruqj


    .. tab-item:: Import / Export

       .. code:: bash

          > bentoml models export iris_clf:latest .

          INFO [cli] Model(tag="iris_clf:2uo5fkgxj27exuqj") exported to ./iris_clf-2uo5fkgxj27exuqj.bentomodel

       .. code:: bash

          > bentoml models import ./iris_clf-2uo5fkgxj27exuqj.bentomodel

          INFO [cli] Model(tag="iris_clf:2uo5fkgxj27exuqj") imported

       .. note::

          Model can be exported to or import from AWS S3, GCS, FTP, Dropbox, etc. For
          example:

          .. code:: bash

             bentoml models export iris_clf:latest s3://my_bucket/my_prefix/

    .. tab-item:: Push / Pull

       If your team has `Yatai <https://github.com/bentoml/Yatai>`_ setup, you can also
       push local Models to Yatai, it provides APIs and Web UI for managing all Models
       created by your team and stores model files on cloud blob storage such as AWS S3,
       MinIO or GCS.

       .. code:: bash

          > bentoml models push iris_clf:latest

          Successfully pushed model "iris_clf:2uo5fkgxj27exuqj"                                                                                                                                                                                           │

       .. code:: bash

          > bentoml models pull iris_clf:latest

          Successfully pulled model "iris_clf:2uo5fkgxj27exuqj"

       .. image:: /_static/img/yatai-model-detail.png
         :alt: Yatai Model Details UI

    .. tab-item:: Delete

       .. code:: bash

          > bentoml models delete iris_clf:latest -y

          INFO [cli] Model(tag="iris_clf:2uo5fkgxj27exuqj") deleted


.. tip::

    Learn more about CLI usage from :code:`bentoml models --help`.


Besides the CLI commands, BentoML also provides equivalent
:doc:`Python APIs </reference/stores>` for managing models:

.. tab-set::

    .. tab-item:: Get

        :code:`bentoml.models.get` returns a :ref:`bentoml.Model </reference/core:Model>`
        instance, which is a reference to a saved model entry in the mdoel store. The
        :code:`bentoml.Model` instances provides access to the model info and the
        :code:`to_runner` API for creating Runner instance:

        .. code:: python

            import bentoml
            bento_model: bentoml.Model = bentoml.models.get("iris_clf:latest")

            print(bento_model.path)
            print(bento_model.info.metadata)
            print(bento_model.info.labels)

            my_runner = bento_model.to_runner()

    .. tab-item:: List

        :code:`bentoml.models.list` returns a list of :ref:`bentoml.Model </reference/core:Model>`:

        .. code:: python

            import bentoml
            models = bentoml.models.list()

    .. tab-item:: Import / Export

        .. code-block:: python

            import bentoml
            bentoml.models.export_model('iris_clf:latest', '/path/to/folder/my_model.bentomodel')

        .. code-block:: python

            bentoml.models.import_model('/path/to/folder/my_model.bentomodel')

        .. note::

            Model can be exported to or import from AWS S3, GCS, FTP, Dropbox, etc. For
            example:

            .. code-block:: python

                bentoml.models.import_model('s3://my_bucket/folder/my_model.bentomodel')


    .. tab-item:: Push / Pull

        If your team has `Yatai <https://github.com/bentoml/Yatai>`_ setup, you can also
        push local Models to Yatai, it provides APIs and Web UI for managing all Models
        created by your team and stores model files on cloud blob storage such as AWS S3,
        MinIO or GCS.

        .. code-block:: python

            import bentoml
            bentoml.models.push("iris_clf:latest")

       .. code-block:: python

            bentoml.models.pull("iris_clf:latest")

       .. image:: /_static/img/yatai-model-detail.png
         :alt: Yatai Model Details UI

    .. tab-item:: Delete

        .. code-block:: python

            import bentoml
            bentoml.models.delete("iris_clf:latest")
