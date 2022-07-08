======
MLflow
======

MLflow for training and experimentation,
BentoML for production serving and deployment,
They work together nicely

Save model natively with BentoML
--------------------------------

    1. save with bentoml next to log_model

        mlflow.sklearn.log_model(..)
        bentoml.sklearn.save_model(..)

    2. load_model(model_uri) and then save with BentoML

        model = mlflow.sklearn.load_model(model_uri)
        bentoml.sklearn.save_model("..", model)

    Note: Recommended using this approach if need GPU inference

Import MLflow model to BentoML
------------------------------

    bentoml.mlflow.import_model("model_name", model_uri=...)

About model_uri:

    URI to the model. A local path, a 'runs:/' URI, or
    a remote storage URI (e.g., an 's3://' URI). For
    more information about supported remote URIs for
    model artifacts, see https://mlflow.org/docs/latest
    /tracking.html#artifact-stores  [required]

    The location, in URI format, of the MLflow model, for example:

    /Users/me/path/to/local/model
    relative/path/to/local/model
    s3://my_bucket/path/to/model
    runs:/<mlflow_run_id>/run-relative/path/to/model
    models:/<model_name>/<model_version>
    models:/<model_name>/<stage>


Load MLflow model saved with BentoML
------------------------------------

* Load with original flavor
    > bento_model = bentoml.mlflow.get("..")
    > pytorch_model = mlflow.pytorch.load_model(bento_model.path_of("mlflow_model"))

* Load pyfunc model
    > bento_model: PyfuncModel = bentoml.mlflow.load_model("..")

* Using MLFlow Runner
    > runner = bentoml.mlflow.get('my_model:latest').to_runner()
    > runner.init_local()
    > runner.predict.run(input_df)
    * supported input output types
        PyFuncInput = Union[pandas.DataFrame, np.ndarray, csc_matrix, csr_matrix, List[Any], Dict[str, Any]]
        PyFuncOutput = Union[pandas.DataFrame, pandas.Series, np.ndarray, list]

        RunnerInput = Union[pandas.DataFrame, np.ndarray, List[Any], Dict[str, Any]]
        RunnerOutput = Union[pandas.DataFrame, pandas.Series, np.ndarray, list]

* Sample service code with MLFlow Runner



MLflow to BentoML workflow
--------------------------

Import from model_uri
~~~~~~~~~~~~~~~~~~~~~
    a. model_info = ...log_model(...), model_info.uri

    b. autolog:
        run_id = mlflow.last_active_run().info.run_id
        model_uri = "runs:/{}/model".format(run_id)

    c. from mlflow.run scope:

        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(lin_reg, "model")
            model_uri = mlflow.get_artifact_uri("model")

            # model_uri = f"runs:/{run.info.run_id}/model"

    d. from mlflow project:
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path="model"
        )


Import a registered model on MLFlow server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    From a version:
        model_name = "sk-learn-random-forest-reg-model"
        model_version = 1
        model_uri=f"models:/{model_name}/{model_version}"
    From a stage:
        model_name = "sk-learn-random-forest-reg-model"
        stage = 'Staging'
        model_uri=f"models:/{model_name}/{stage}"

    note: how to verify model_uri can be accessed

Import from Databricks MLFlow Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    See Databricks integration guide


Using MLFlow model dependency in Bento
--------------------------------------

    Using MLFlow model's python dependencies

        python:
            requirements_txt: $BENTOML_MLFLOW_MODEL_PATH/mlflow_model/requirements.txt
            lock_packages: False

    Alternatively use MLFlow model's conda environment

        conda:
            environment_yml: $BENTOML_MLFLOW_MODEL_PATH/mlflow_model/conda.yaml

    export BENTOML_MLFLOW_MODEL_PATH=$(bentoml models get iris_clf:latest -o path)
    bentoml build

Import MLFLow model with run metrics and tags
---------------------------------------------

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
    run.info.artifact_uri