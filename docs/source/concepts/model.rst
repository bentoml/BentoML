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

    saved_model = bentoml.sklearn.save_model("iris_clf", clf)
    print(f"Model saved: {saved_model}")

    # Model saved: Model(tag="iris_clf:2uo5fkgxj27exuqj")

.. seealso::

   It is also possible to **use pre-trained models** directly with BentoML, without
   saving it to the model store first. Check out the
   :ref:`Custom Runner <concepts/runner:Custom Runner>` example to learn more.

.. tip::

   If you have an existing model saved to file on disk, you will need to load the model
   in a python session first and then use BentoML's framework specific
   :code:`save_model` method to put it into the BentoML model store.

   We recommend **always save the model with BentoML as soon as it finished training and
   validation**. By putting the :code:`save_model` call to the end of your training
   pipeline, all your finalized models can be managed in one place and ready for
   inference.


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


Retrieve a saved model
----------------------

To load the model instance back into memory, use the framework-specific
:code:`load_model` method. For example:

.. code:: python

    import bentoml
    from sklearn.base import BaseEstimator

    model: BaseEstimator = bentoml.sklearn.load_model("iris_clf:latest")

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

:code:`bentoml.models.get` returns a :ref:`bentoml.Model <reference/core:Model>`
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

    Model(tag="iris_clf:2uo5fkgxj27exuqj") exported to ./iris_clf-2uo5fkgxj27exuqj.bentomodel

.. code:: bash

    > bentoml models import ./iris_clf-2uo5fkgxj27exuqj.bentomodel

    Model(tag="iris_clf:2uo5fkgxj27exuqj") imported

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

        :code:`bentoml.models.list` returns a list of :ref:`bentoml.Model <reference/core:Model>`:

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

The way to run model inference in the context of a :code:`bentoml.Service`, is via a
Runner. The Runner abstraction gives BentoServer more flexibility in terms of how to
schedule the inference computation, how to dynamically batch inference calls and better
take advantage of all hardware resource available.

As demonstrated in the :doc:`tutorial </tutorial>`, a model runner can be created
from a saved model via the :code:`to_runner` API:

.. code:: python

    iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()


The runner instance can then be used for creating a :code:`bentoml.Service`:

.. code:: python

    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        result = iris_clf_runner.predict.run(input_series)
        return result


To test out the runner interface before writing the Service API callback function,
you can create a local runner instance outside of a Service:

.. code:: python

    # Create a Runner instance:
    iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

    # Initializes the runner in current process, this is meant for development and testing only:
    iris_clf_runner.init_local()

    # This should yield the same result as the loaded model:
    iris_clf_runner.predict.run([[5.9, 3., 5.1, 1.8]])


To learn more about Runner usage and its architecture, see :doc:`/concepts/runner`.


Model Signatures
----------------

A model signature represents a method on a model object that can be called. This
information is used when creating BentoML runners for this model.

From the example above, the :code:`iris_clf_runner.predict.run` call will pass through
the function input to the model's :code:`predict` method, running from a remote runner
process.

For many :doc:`other ML frameworks </frameworks/index>`, the model object's inference
method may not be called :code:`predict`. Users can customize it by specifying the model
signature during :code:`save_model`:

.. code-block:: python
   :emphasize-lines: 4-8,13

    bentoml.pytorch.save_model(
        "demo_mnist",  # model name in the local model store
        trained_model,  # model instance being saved
        signatures={   # model signatures for runner inference
            "classify": {
                "batchable": False,
            }
        }
    )

    runner = bentoml.pytorch.get("demo_mnist:latest").to_runner()
    runner.init_local()
    runner.classify.run( MODEL_INPUT )


A special case here is Python's magic method :code:`__call__`. Similar to the
Python language convention, the call to :code:`runner.run` will be applied to
the model's :code:`__call__` method:

.. code-block:: python
   :emphasize-lines: 4-8,13

    bentoml.pytorch.save_model(
        "demo_mnist",  # model name in the local model store
        trained_model,  # model instance being saved
        signatures={   # model signatures for runner inference
            "__call__": {
                "batchable": False,
            },
        }
    )

    runner = bentoml.pytorch.get("demo_mnist:latest").to_runner()
    runner.init_local()
    runner.run( MODEL_INPUT )

Batching
--------

For model inference calls that supports taking a batch input, it is recommended to
enable batching for the target model signature. In which case, :code:`runner#run` calls
made from multiple Service workers can be dynamically merged to a larger batch and run
as one inference call in the runner worker. Here's an example:

.. code-block:: python
   :emphasize-lines: 4-9,14

    bentoml.pytorch.save_model(
        "demo_mnist",  # model name in the local model store
        trained_model,  # model instance being saved
        signatures={   # model signatures for runner inference
            "__call__": {
                "batchable": True,
                "batch_dim": 0,
            },
        }
    )

    runner = bentoml.pytorch.get("demo_mnist:latest").to_runner()
    runner.init_local()
    runner.run( MODEL_INPUT )

.. tip::

    The runner interface is exactly the same, regardless :code:`batchable` was set to
    True or False.

The :code:`batch_dim` parameter determines the dimension(s) that contain multiple data
when passing to this run method. The default :code:`batch_dim`, when left unspecified,
is :code:`0`.

For example, if you have two inputs you want to run prediction on, :code:`[1, 2]` and
:code:`[3, 4]`, if the array you would pass to the predict method would be
:code:`[[1, 2], [3, 4]]`, then the batch dimension would be :code:`0`. If the array you
would pass to the predict method would be :code:`[[1, 3], [2, 4]]`, then the batch
dimension would be :code:`1`. For example:

.. code:: python

    # Save two models with `predict` method that supports taking input batches on the
    # dimension 0 and the other on dimension 1:
    bentoml.pytorch.save_model("demo0", model_0, signatures={
        "predict": {"batchable": True, "batch_dim": 0}}
    )
    bentoml.pytorch.save_model("demo1", model_1, signatures={
        "predict": {"batchable": True, "batch_dim": 1}}
    )

    # if the following calls are batched, the input to the actual predict method on the
    # model.predict method would be [[1, 2], [3, 4], [5, 6]]
    runner0 = bentoml.pytorch.get("demo0:latest").to_runner()
    runner0.init_local()
    runner0.predict.run(np.array([[1, 2], [3, 4]]))
    runner0.predict.run(np.array([[5, 6]]))

    # if the following calls are batched, the input to the actual predict method on the
    # model.predict would be [[1, 2, 5], [3, 4, 6]]
    runner1 = bentoml.pytorch.get("demo1:latest").to_runner()
    runner1.init_local()
    runner1.predict.run(np.array([[1, 2], [3, 4]]))
    runner1.predict.run(np.array([[5], [6]]))


.. admonition:: Expert API

    If there are multiple arguments to the run method and there is only one batch
    dimension supplied, all arguments will use that batch dimension.

    The batch dimension can also be a tuple of (input batch dimension, output batch
    dimension). For example, if the predict method should have its input batched along
    the first axis and its output batched along the zeroth axis, :code:`batch_dim`` can
    be set to :code:`(1, 0)`.


For online serving workloads, adaptive batching is a critical component that contributes
to the overall performance. If throughput and latency are important to you, learn more
about other Runner options and batching configurations in the :doc:`/concepts/runner`
and :doc:`/guides/batching` doc.


.. TODO::
    Add example for using ModelOptions for setting runtime options
