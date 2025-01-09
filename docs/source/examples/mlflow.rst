======
MLflow
======

`MLflow <https://mlflow.org/>`_ is an open-source platform, purpose-built to assist machine learning practitioners and teams in handling the complexities of the machine learning process. MLflow focuses on the full lifecycle for machine learning projects, ensuring that each phase is manageable, traceable, and reproducible.

This document explains how to serve and deploy an MLflow model with BentoML. You can find all the source code `here <https://github.com/bentoml/BentoMLflow>`_.

Prerequisites
-------------

- Python 3.9+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read :doc:`/get-started/hello-world` first.
- (Optional) We recommend you create a virtual environment for dependency isolation. See the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or the `Python documentation <https://docs.python.org/3/library/venv.html>`_ for details.

Install dependencies
--------------------

.. code-block:: bash

    pip install bentoml scikit-learn mlflow

Train and save a model
----------------------

This example uses the ``scikit-learn`` framework to train a classification model and saves it with MLflow.

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier
    from pathlib import Path
    import mlflow.sklearn

    iris = load_iris()
    X_train = iris.data[:, :4]
    Y_train = iris.target

    model_uri = Path("models", "IrisClf")
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    mlflow.sklearn.save_model(model, model_uri.resolve())

Next, use the ``bentoml.mlflow.import_model`` API to save the model to the BentoML :doc:`Model Store </build-with-bentoml/model-loading-and-management>`, a local directory to store and manage models. You can retrieve this model later in other services to run predictions.

.. code-block:: bash

    import bentoml

    # model_uri can be any URI that refers to an MLflow model
    # Use local path for demostration
    bentoml.mlflow.import_model("iris", model_uri)

To verify that the model has been successfully saved, run:

.. code-block:: bash

    $ bentoml models list

    Tag                      Module           Size       Creation Time
    iris:74px7hboeo25fjjt    bentoml.mlflow   10.07 KiB  2024-06-19 10:09:21

Test the saved model
--------------------

To ensure that the saved model works correctly, try loading it and running a prediction:

.. code-block:: python

    import numpy as np
    import bentoml

    # Load the model by specifying the model tag
    iris_model = bentoml.mlflow.load_model("iris:74px7hboeo25fjjt")

    input_data = np.array([[5.9, 3, 5.1, 1.8]])
    res = iris_model.predict(input_data)
    print(res)

Expected result:

.. code-block:: bash

    [2] # The model thinks the category seems to be Virginica.

Create a BentoML Service
------------------------

Create a separate ``service.py`` file where you define a BentoML :doc:`Service </build-with-bentoml/services>` to expose the model as a web service.

.. code-block:: python

    import bentoml
    import numpy as np

    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
    )
    class IrisClassifier:
        bento_model = bentoml.models.get("iris:latest")

        def __init__(self):
            self.model = bentoml.mlflow.load_model(self.bento_model)

        @bentoml.api
        def predict(self, input_data: np.ndarray) -> np.ndarray:
            rv = self.model.predict(input_data)
            return np.asarray(rv)

The Service code:

- Uses the ``@bentoml.service`` decorator to define a BentoML Service. Optionally, you can set additional :doc:`configurations </reference/bentoml/configurations>` like resource allocation and traffic timeout.
- Retrieves the model from the Model Store and defines it a class variable.
- Uses the ``@bentoml.api`` decorator to expose the ``predict`` function as an API endpoint, which :doc:`takes a NumPy array as input and returns a NumPy array </build-with-bentoml/iotypes>`.

Run ``bentoml serve`` in your project directory to start the Service.

.. code-block:: bash

    $ bentoml serve service:IrisClassifier

    2024-06-19T10:25:31+0000 [WARNING] [cli] Converting 'IrisClassifier' to lowercase: 'irisclassifier'.
    2024-06-19T10:25:31+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:IrisClassifier" listening on http://localhost:3000 (Press CTRL+C to quit)

The server is active at `http://localhost:3000 <http://localhost:3000/>`_. You can interact with it in different ways.

.. tab-set::

    .. tab-item:: CURL

        .. code-block:: bash

            curl -X 'POST' \
                'http://localhost:3000/predict' \
                -H 'accept: application/json' \
                -H 'Content-Type: application/json' \
                -d '{
                "input_data": [
                    [5.9, 3, 5.1, 1.8]
                ]
            }'

    .. tab-item:: Python client

        .. code-block:: python

            import bentoml

            with bentoml.SyncHTTPClient("http://localhost:3000") as client:
                result = client.predict(
                    input_data=[
                        [5.9, 3, 5.1, 1.8]
                    ],
                )
                print(result)

    .. tab-item:: Swagger UI

        Visit `http://localhost:3000 <http://localhost:3000/>`_, scroll down to **Service APIs**, specify the data, and click **Execute**.

        .. image:: ../../_static/img/examples/mlflow/service-ui.png

Deploy to BentoCloud
--------------------

After the Service is ready, you can deploy it to BentoCloud for better management and scalability. `Sign up <https://www.bentoml.com/>`_ for a BentoCloud account and get $10 in free credits.

Specify a configuration YAML file (``bentofile.yaml``) to define the build options for a :doc:`Bento </reference/bentoml/bento-build-options>`, the unified distribution format in BentoML containing source code, Python packages, model references, and so on. Here is an example file:

.. code-block:: yaml

    service: "service:IrisClassifier"
    labels:
      owner: bentoml-team
      stage: demo
    include:
      - "*.py"
    python:
      packages:
        - mlflow
        - scikit-learn

:ref:`Log in to BentoCloud <scale-with-bentocloud/manage-api-tokens:Log in to BentoCloud using the BentoML CLI>` by running ``bentoml cloud login``, then run the following command to deploy the project.

.. code-block:: bash

    bentoml deploy .

Once the Deployment is up and running on BentoCloud, you can access it via the exposed URL.

.. image:: ../../_static/img/examples/mlflow/bentocloud-ui.png

.. note::

   For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
