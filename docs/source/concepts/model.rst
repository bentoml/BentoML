================
Preparing Models
================

Save A Trained Model
--------------------

A trained ML model instance needs to be saved with BentoML API, in order to serve it
with BentoML. For most cases, it will be just one line added to your model training
pipeline, invoking a :code:`save_model` call, as demonstrated in the
:doc:`tutorial </tutorial>`:

.. code:: python

    bentoml.sklearn.save_model("iris_clf", clf)

    # INFO  [cli] Using default model signature `{"predict": {"batchable": False}}` for sklearn model
    # INFO  [cli] Successfully saved Model(tag="iris_clf:2uo5fkgxj27exuqj", path="~/bentoml/models/iris_clf/2uo5fkgxj27exuqj/")

Optionally, you may attach custom labels, metadata, or :code:`custom_objects` to be
saved alongside your model in the model store, e.g.:

.. code:: python

    bentoml.pytorch.save_model(
        "demo_mnist",   # model name in the local model store
        trained_model,  # model instance being saved
        labels={    # user-defined labels for managing models in Yatai
            "owner": "nlp_team",
            "stage": "dev",
        },
        metadata={  # user-defined additional metadata
            "acc": acc,
            "cv_stats": cv_stats,
            "dataset_version": "20210820",
        },
        custom_objects={    # save additional user-defined python objects
            "tokenizer": tokenizer_object,
        }
    )

- **labels**: user-defined labels for managing models, e.g. team=nlp, stage=dev.
- **metadata**: user-defined metadata for storing model training context information or model evaluation metrics, e.g. dataset version, training parameters, model scores.
- **custom_objects**: user-defined additional python objects, e.g. a tokenizer instance, preprocessor function, model configuration json, serialized with cloudpickle. Custom objects will be serialized with `cloudpickle <https://github.com/cloudpipe/cloudpickle>`_.


.. TODO::
    Add example for using ModelOptions


Retrieve a saved model
----------------------

To load the model instance back into memory, use the framework-specific
:code:`load_model` method. For example:

.. code:: python

    import bentoml
    from sklearn.base import BaseEstimator

    model:BaseEstimator = bentoml.sklearn.load_model("iris_clf:latest")

.. note::
    The :code:`load_model` method is only here for testing and advanced customizations.
    For general model serving use cases, use Runner for running model inference. See the
    :ref:`concepts/model:Using Model Runner` section below to learn more.

For retrieving the model metadata or custom objects, use the :code:`get` method:

.. code:: python

    import bentoml
    bento_model: bentoml.Model = bentoml.models.get("iris_clf:latest")

    print(bento_model.tag)
    print(bento_model.path)
    print(bento_model.custom_objects)
    print(bento_model.info.metadata)
    print(bento_model.info.labels)

    my_runner: bentoml.Runner = bento_model.to_runner()

:code:`bentoml.models.get` returns a :ref:`bentoml.Model </reference/core:Model>`
instance, which is a reference to a saved model entry in the BentoML model store. The
:code:`bentoml.Model` instance then provides access to the model info and the
:code:`to_runner` API for creating a Runner instance from the model.

.. note::
    BentoML also provides a framework-specific :code:`get` method under each framework
    module, e.g.: :code:`benotml.pytorch.get`. It behaves exactly the same as
    :code:`bentoml.models.get`, besides that it verifies if the model found was saved
    with the same framework.


Managing Models
---------------

Saved models are stored in BentoML's model store, which is a local file directory
maintained by BentoML. Users can view and manage all saved models via the
:code:`bentoml models` CLI command:

.. tab-set::

    .. tab-item:: List

        .. code:: bash

            > bentoml models list

            Tag                        Module           Size        Creation Time        Path
            iris_clf:2uo5fkgxj27exuqj  bentoml.sklearn  5.81 KiB    2022-05-19 08:36:52  ~/bentoml/models/iris_clf/2uo5fkgxj27exuqj
            iris_clf:nb5vrfgwfgtjruqj  bentoml.sklearn  5.80 KiB    2022-05-17 21:36:27  ~/bentoml/models/iris_clf/nb5vrfgwfgtjruqj


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

    .. tab-item:: Delete

        .. code:: bash

            > bentoml models delete iris_clf:latest -y

            INFO [cli] Model(tag="iris_clf:2uo5fkgxj27exuqj") deleted



Model Import and Export
^^^^^^^^^^^^^^^^^^^^^^^

Models saved with BentoML can be exported to a standalone archive file outside of the
model store, for sharing models between teams or moving models between different build
stages. For example:

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

        pip install fs-s3fs  # Additional dependency required for working with s3
        bentoml models export iris_clf:latest s3://my_bucket/my_prefix/


Push and Pull with Yatai
^^^^^^^^^^^^^^^^^^^^^^^^

`Yatai <https://github.com/bentoml/Yatai>`_ provides a centralized Model repository
that comes with flexible APIs and Web UI for managing all models (and
:doc:`Bentos </concepts/bento>`) created by your team. It can be configured to store
model files on cloud blob storage such as AWS S3, MinIO or GCS.

Once your team have Yatai setup, you can use the :code:`bentoml models push` and
:code:`bentoml models pull` command to get models to and from Yatai:

.. code:: bash

    > bentoml models push iris_clf:latest

    Successfully pushed model "iris_clf:2uo5fkgxj27exuqj"                                                                                                                                                                                           │

.. code:: bash

    > bentoml models pull iris_clf:latest

    Successfully pulled model "iris_clf:2uo5fkgxj27exuqj"

.. image:: /_static/img/yatai-model-detail.png
    :alt: Yatai Model Details UI


.. tip::

    Learn more about CLI usage from :code:`bentoml models --help`.


Model Management API
^^^^^^^^^^^^^^^^^^^^

Besides the CLI commands, BentoML also provides equivalent
:doc:`Python APIs </reference/stores>` for managing models:

.. tab-set::

    .. tab-item:: Get

        .. code:: python

            import bentoml
            bento_model: bentoml.Model = bentoml.models.get("iris_clf:latest")

            print(bento_model.path)
            print(bento_model.info.metadata)
            print(bento_model.info.labels)


    .. tab-item:: List

        :code:`bentoml.models.list` returns a list of :ref:`bentoml.Model </reference/core:Model>`:

        .. code:: python

            import bentoml
            models = bentoml.models.list()

    .. tab-item:: Import / Export

        .. code:: python

            import bentoml
            bentoml.models.export_model('iris_clf:latest', '/path/to/folder/my_model.bentomodel')

        .. code:: python

            bentoml.models.import_model('/path/to/folder/my_model.bentomodel')

        .. note::

            Model can be exported to or import from AWS S3, GCS, FTP, Dropbox, etc. For
            example:

            .. code:: python

                bentoml.models.import_model('s3://my_bucket/folder/my_model.bentomodel')


    .. tab-item:: Push / Pull

        If your team has `Yatai <https://github.com/bentoml/Yatai>`_ setup, you can also
        push local Models to Yatai, it provides APIs and Web UI for managing all Models
        created by your team and stores model files on cloud blob storage such as AWS S3,
        MinIO or GCS.

        .. code:: python

            import bentoml
            bentoml.models.push("iris_clf:latest")

        .. code:: python

            bentoml.models.pull("iris_clf:latest")


    .. tab-item:: Delete

        .. code:: python

            import bentoml
            bentoml.models.delete("iris_clf:latest")


Using Model Runner
------------------

The :doc:`tutorial </tutorial>`

.. code:: python

    iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        result = iris_clf_runner.predict.run(input_series)
        return result

.. code:: python

  @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
  async def classify(input_series: np.ndarray) -> np.ndarray:
     result = await iris_clf_runner.predict.async_run(input_series)
     return result


.. code:: python

    # Create a Runner instance:
    iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

    # Runner#init_local initializes the model in current process, this is meant for development and testing only:
    iris_clf_runner.init_local()

    # This should yield the same result as the loaded model:
    iris_clf_runner.predict.run([[5.9, 3., 5.1, 1.8]])


To learn more about Runner usage and its architecture, see :doc:`/concepts/runner`.


Model Signatures and Batching
-----------------------------

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



get, to_runner
testing runner
runner input/output

Model signature
* batchable
* batch_dim

Dynamic batching params
* max_batch_size
* max_latency_ms

