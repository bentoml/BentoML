======
MLflow
======

`MLflow <https://mlflow.org/>`_ is an open source framework for tracking ML experiments,
packaging ML code for training pipelines, and capturing models logged from experiments.
It enables data scientists to iterate quickly during model development while keeping
their experiments and training pipelines reproducible.

BentoML, on the other hand, focuses on ML in production. By design, BentoML is agnostic
to the experimentation platform and the model development environment.

Comparing to the MLflow model registry, BentoML's model format and model store is
designed for managing model artifacts that will be used for building, testing, and
deploying prediction services. It is best fitted to manage your “finalized model”, sets
of models that yield the best outcomes from your periodic training pipelines and are
meant for running in production.

BentoML integrates with MLflow natively. Users can not only port over models logged with
MLflow Tracking to BentoML for high-performance model serving but also combine MLFlow
projects and pipelines with BentoML's model deployment workflow in an efficient manner.


Compatibility
-------------

BentoML supports MLflow 0.9 and above.

Examples
--------

Besides this documentation, also check out code samples demonstrating BentoML and MLflow
integration at: `bentoml/examples: MLflow Examples <https://github.com/bentoml/BentoML/tree/main/examples/mlflow>`_.


Import an MLflow model
----------------------

`MLflow Model <https://www.mlflow.org/docs/latest/models.html>`_ is a format for saving
trained model artifacts in MLflow experiments and pipelines. BentoML supports importing
MLflow model to its own format for model serving. For example:

.. code-block:: python

    mlflow.sklearn.save_model(model, "./my_model")
    bentoml.mlflow.import_model("my_sklearn_model", model_uri="./my_model")


.. code-block:: python

    with mlflow.start_run():
        mlflow.pytorch.log_model(model, artifact_path="pytorch-model")

        model_uri = mlflow.get_artifact_uri("pytorch-model")
        bento_model = bentoml.mlflow.import_model(
            'mlflow_pytorch_mnist',
            model_uri,
            signatures={'predict': {'batchable': True}}
        )


The ``bentoml.mlflow.import_model`` API is similar to the other ``save_model`` APIs
found in BentoML, where the first argument represent the model name in BentoML model
store. A new version will be automatically generated when a new MLflow model is
imported. Users can manage imported MLflow models same as models saved with other ML
frameworks:

.. code-block:: bash

    bentoml models list mlflow_pytorch_mnist


The second argument ``model_uri`` takes a URI to the MLflow model. It can be a local
path, a ``'runs:/'`` URI, or a remote storage URI (e.g., an ``'s3://'`` URI). Here are
some example ``model_uri`` values commonly used in MLflow:

.. code-block::

    /Users/me/path/to/local/model
    ../relative/path/to/local/model
    s3://my_bucket/path/to/model
    runs:/<mlflow_run_id>/run-relative/path/to/model
    models:/<model_name>/<model_version>
    models:/<model_name>/<stage>


Running Imported Model
----------------------

MLflow models imported to BentoML can be loaded back for running inference in a various
of ways.

Loading original model flavor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For evaluation and testing purpose, sometimes it's convenient to load the model in its
native form

.. code-block:: python

    bento_model = bentoml.mlflow.get("mlflow_pytorch_mnist:latest")
    mlflow_model_path = bento_model.path_of(bentoml.mlflow.MLFLOW_MODEL_FOLDER)

    loaded_pytorch_model = mlflow.pytorch.load_model(mlflow_model_path)
    loaded_pytorch_model.to(device)
    loaded_pytorch_model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(test_input_arr).to(device)
        predictions = loaded_pytorch_model(input_tensor)


Loading Pyfunc flavor
~~~~~~~~~~~~~~~~~~~~~

By default, ``bentoml.mflow.load_model`` will load the imported MLflow model using the
`python_function flavor <https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html>`_
for best compatibility across all ML frameworks supported by MLflow.

.. code-block:: python

    pyfunc_model: mlflow.pyfunc.PyFuncModel = bentoml.mlflow.load_model("mlflow_pytorch_mnist:latest")
    predictions = pyfunc_model.predict(test_input_arr)


Using Model Runner
~~~~~~~~~~~~~~~~~~

Imported MLflow models can be loaded as BentoML Runner for best performance in building
prediction service with BentoML. To test out the runner API:

.. code-block:: python

    runner = bentoml.mlflow.get("mlflow_pytorch_mnist:latest").to_runner()
    runner.init_local()
    runner.predict.run(input_df)

Learn more about BentoML Runner at :doc:`/concepts/runner`.

Runner created from an MLflow model supports the following input types. Note that for
some ML frameworks, only a subset of this list is supported.

.. code-block:: python

    MLflowRunnerInput = Union[pandas.DataFrame, np.ndarray, List[Any], Dict[str, Any]]
    MLflowRunnerOutput = Union[pandas.DataFrame, pandas.Series, np.ndarray, list]

.. note::

    To use adaptive batching with a MLflow Runner, make sure to set
    ``signatures={'predict': {'batchable': True}}`` when importing the model:

    .. code-block:: python

        bento_model = bentoml.mlflow.import_model(
            'mlflow_pytorch_mnist',
            model_uri,
            signatures={'predict': {'batchable': True}}
        )


Optimizations
~~~~~~~~~~~~~

There are two major limitations of using MLflow Runner in BentoML:

* Lack of support for GPU
* Lack of support for multiple inference method

A common optimization we recommend, is to save trained model instance directly with BentoML,
instead of importing MLflow pyfunc model. This makes it possible to run GPU inference and expose 
multiple inference signatures.

1. Save model directly with bentoml

.. code-block:: python

    mlflow.sklearn.log_model(clf, "model")
    bentoml.sklearn.save_model("iris_clf", clf)

2. Load original flavor and save with BentoML

.. code-block:: python

    loaded_model = mlflow.sklearn.load_model(model_uri)
    bentoml.sklearn.save_model("iris_clf", loaded_model)

This way, it goes back to a typically BentoML workflow, which allow users to use a
Runner specifically built for the target ML framework, with GPU support and multiple
signatures available.


Build Prediction Service
------------------------

Here's an example ``bentoml.Service`` built with a MLflow model:

.. code-block:: python

    import bentoml
    import mlflow
    import torch

    mnist_runner = bentoml.mlflow.get('mlflow_pytorch_mnist:latest').to_runner()

    svc = bentoml.Service('mlflow_pytorch_mnist', runners=[ mnist_runner ])

    input_spec = bentoml.io.NumpyNdarray(
        dtype="float32",
        shape=[-1, 1, 28, 28],
        enforce_shape=True,
        enforce_dtype=True,
    )

    @svc.api(input=input_spec, output=bentoml.io.NumpyNdarray())
    def predict(input_arr):
        return mnist_runner.predict.run(input_arr)

To try out the full example, visit `bentoml/examples: MLflow Pytorch Example <https://github.com/bentoml/BentoML/tree/main/examples/mlflow/pytorch>`_.


MLflow 🤝 BentoML Workflow
--------------------------

There are numerous ways you can integrate BentoML with your MLflow workflow for model serving and deployment.

1. Find ``model_uri`` from a MLflow model instance returned from ``log_model``:

.. code-block:: python

    # https://github.com/bentoml/BentoML/tree/main/examples/mlflow/sklearn_logistic_regression
    logged_model = mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    # Import logged mlflow model to BentoML model store for serving:
    bento_model = bentoml.mlflow.import_model('logistic_regression_model', logged_model.model_uri)
    print("Model imported to BentoML: %s" % bento_model)

2. Find model artifact path inside current ``mlflow.run`` scope:

.. code-block:: python

    # https://github.com/bentoml/BentoML/tree/main/examples/mlflow/pytorch
    with mlflow.start_run():
        ...
        mlflow.pytorch.log_model(model, artifact_path="pytorch-model")
        model_uri = mlflow.get_artifact_uri("pytorch-model")
        bento_model = bentoml.mlflow.import_model('mlflow_pytorch_mnist', model_uri)

3. When using ``autolog``, find ``model_uri`` by last active ``run_id``:

.. code-block:: python

    import mlflow
    import bentoml
    from sklearn.linear_model import LinearRegression

    # enable autologging
    mlflow.sklearn.autolog()

    # prepare training data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3

    # train a model
    model = LinearRegression()
    model.fit(X, y)

    # import logged MLflow model to BentoML
    run_id = mlflow.last_active_run().info.run_id
    artifact_path = "model"
    model_uri = f"runs:/{run_id}/{artifact_path}"
    bento_model = bentoml.mlflow.import_model('logistic_regression_model', model_uri)
    print(f"Model imported to BentoML: {bento_model}")



4. Import a registered model on MLflow server

When using a MLflow tracking server, users can also import
`registered models <https://www.mlflow.org/docs/latest/model-registry.html#registering-a-model>`_
directly to BentoML for serving.

.. code-block:: python

    # Import from a version:
    model_name = "sk-learn-random-forest-reg-model"
    model_version = 1
    model_uri=f"models:/{model_name}/{model_version}"
    bentoml.mlflow.import_model('my_mlflow_model', model_uri)

    # Import from a stage:
    model_name = "sk-learn-random-forest-reg-model"
    stage = 'Staging'
    model_uri=f"models:/{model_name}/{stage}"
    bentoml.mlflow.import_model('my_mlflow_model', model_uri)


Additional Tips
---------------

Use MLflow model dependencies config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most MLflow models bundles dependency information that is required for running framework model. 
If no additional dependencies are required in the :obj:`~bentoml.Service` definition code, users may
pass through dependency requirements from within MLflow model to BentoML.

First, put the following in your ``bentofile.yaml`` build file:

.. code-block:: yaml

    python:
        requirements_txt: $BENTOML_MLFLOW_MODEL_PATH/mlflow_model/requirements.txt
        lock_packages: False

Alternatively, one can also use MLflow model's generated conda environment file:

.. code-block:: yaml

    conda:
        environment_yml: $BENTOML_MLFLOW_MODEL_PATH/mlflow_model/conda.yaml

This allows BentoML to dynamically find the given dependency file based on a user-defined
environment variable. In this case, the ``bentoml get`` CLI returns the path to the target
MLflow model folder and expose it to ``bentoml build`` via the environment variable
``BENTOML_MLFLOW_MODEL_PATH``:

.. code-block:: bash

    export BENTOML_MLFLOW_MODEL_PATH=$(bentoml models get my_mlflow_model:latest -o path)
    bentoml build


Attach model params, metrics, and tags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MLflow model format encapsulates lots of context information regarding the training metrics
and parameters. The following code snippet demonstrates how to package metadata logged from a given MLflow model to the BentoML model store.


.. code-block:: python

    run_id = '0e4425ecbf3e4672ba0c1741651bb47a'
    run = mlflow.get_run(run_id)
    model_uri = f"{run.info.artifact_uri}/model"
    bentoml.mlflow.import_model(
        "my_mlflow_model",
        model_uri,
        labels=run.data.tags,
        metadata={
            "metrics": run.data.metrics,
            "params": run.data.params,
        }
    )
