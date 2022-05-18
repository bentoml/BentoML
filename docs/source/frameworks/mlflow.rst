======
MLFlow
======

Users can now use MLFlow with BentoML with the following API: :code:`load`, and :code:`load_runner` as follow:

.. code-block:: python

   import bentoml
   import mlflow
   import pandas as pd

   # `load` the model back in memory:
   model = bentoml.mlflow.load("mlflow_sklearn_model:latest")
   model.predict(pd.DataFrame[[1,2,3]])

   # Load a given tag and run it under `Runner` abstraction with `load_runner`
   runner = bentoml.mlflow.load_runner(tag)
   runner.run_batch([[1,2,3,4,5]])

BentoML also offer :code:`import_from_uri` which enables users to import any MLFlow model to BentoML:

.. code-block:: python

    import bentoml
    from pathlib import Path

    # assume that there is a folder name sklearn_clf in the current working directory
    uri = Path("sklearn_clf").resolve()
    
    # get the given tag
    tag = bentoml.mlflow.import_from_uri("sklearn_clf_model", uri)

    # uri can also be a S3 bucket
    s3_tag = bentoml.mlflow.import_from_uri("sklearn_clf_model", "s3://my_sklearn_model")

.. note::

   You can find more examples for **MLFlow** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.mlflow

.. autofunction:: bentoml.mlflow.import_from_uri

.. autofunction:: bentoml.mlflow.load

.. autofunction:: bentoml.mlflow.load_runner

.. autofunction:: bentoml.mlflow.save
