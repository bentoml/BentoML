======
MLflow
======

`MLflow <https://mlflow.org/>`_ is an open-source platform, purpose-built to assist machine learning practitioners and teams in handling the complexities of the machine learning process. MLflow focuses on the full lifecycle for machine learning projects, ensuring that each phase is manageable, traceable, and reproducible.

This document explains how to serve and deploy an MLflow model with BentoML.

.. raw:: html

    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; flex-grow: 1; margin-right: 10px; text-align: center;">
            <img src="https://docs.bentoml.com/en/latest/_static/img/github-mark.png" alt="GitHub" style="vertical-align: middle; width: 24px; height: 24px;">
            <a href="https://github.com/bentoml/BentoMLflow" style="margin-left: 5px; vertical-align: middle;">Source Code</a>
        </div>
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; flex-grow: 1; margin-left: 10px; text-align: center;">
            <img src="https://docs.bentoml.com/en/latest/_static/img/bentocloud-logo.png" alt="BentoCloud" style="vertical-align: middle; width: 24px; height: 24px;">
            <a href="#bentocloud" style="margin-left: 5px; vertical-align: middle;">Deploy to BentoCloud</a>
        </div>
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; flex-grow: 1; margin-left: 10px; text-align: center;">
            <img src="https://docs.bentoml.com/en/latest/_static/img/bentoml-icon.png" alt="BentoML" style="vertical-align: middle; width: 24px; height: 24px;">
            <a href="#localserving" style="margin-left: 5px; vertical-align: middle;">Serve with BentoML</a>
        </div>
    </div>

The example uses scikit-learn for demo purposes. You can submit a classification request to the endpoint like this:

.. code-block:: bash

   {
        "input_data": [[5.9,3,5.1,1.8]]
   }

Expected output:

.. code-block:: bash

   ["virginica"]

In addition to scikit-learn, both MLflow and BentoML support a wide variety of other frameworks, such as PyTorch, TensorFlow and XGBoost.

This example is ready for quick deployment and scaling on BentoCloud. With a single command, you get a production-grade application with fast autoscaling, secure deployment in your cloud, and comprehensive observability.

.. image:: ../../_static/img/examples/mlflow/mlflow-model-on-bentocloud.png
    :alt: Screenshot of MLflow Iris classifier model deployed on BentoCloud showing the prediction interface with input data and classification results

Code explanations
-----------------

You can find `the source code in GitHub <https://github.com/bentoml/BentoMLflow>`_. Below is a breakdown of the key code implementations within this project.

save_model.py
^^^^^^^^^^^^^

This example uses the ``scikit-learn`` framework to train a classification model and saves it with MLflow.

.. code-block:: python
    :caption: `save_model.py`

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

After training, use the ``bentoml.mlflow.import_model`` API to save the model to the BentoML :doc:`Model Store </build-with-bentoml/model-loading-and-management>`, a local directory to store and manage models. You can retrieve this model later in other services to run predictions.

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

test.py
^^^^^^^

To ensure that the saved model works correctly, try loading it and running a prediction:

.. code-block:: python
    :caption: `test.py`

    import numpy as np
    import bentoml

    # Load the latest version of iris model
    iris_model = bentoml.mlflow.load_model("iris:latest")

    input_data = np.array([[5.9, 3, 5.1, 1.8]])
    res = iris_model.predict(input_data)
    print(res)

Expected result:

.. code-block:: bash

    [2] # The model thinks the category seems to be Virginica.

service.py
^^^^^^^^^^

The ``service.py`` file is where you define the serving logic and expose the model as a web service.

.. code-block:: python
    :caption: `service.py`

    import bentoml
    import numpy as np

    target_names = ['setosa', 'versicolor', 'virginica']

    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
    )
    class IrisClassifier:
        # Declare the model as a class variable
        bento_model = bentoml.models.BentoModel("iris:latest")

        def __init__(self):
            self.model = bentoml.mlflow.load_model(self.bento_model)

        @bentoml.api
        def predict(self, input_data: np.ndarray) -> list[str]:
            preds = self.model.predict(input_data)
            return [target_names[i] for i in preds]

The Service code:

- Uses the ``@bentoml.service`` decorator to define a BentoML :doc:`Service </build-with-bentoml/services>`. Optionally, you can set additional :doc:`configurations </reference/bentoml/configurations>` like resource allocation on BentoCloud and traffic timeout.
- Retrieves the model from the Model Store and defines it a class variable.
- Uses the ``@bentoml.api`` decorator to expose the ``predict`` function as an API endpoint.

The ``@bentoml.service`` decorator also allows you to :doc:`define the runtime environment </build-with-bentoml/runtime-environment>` for a Bento, the unified distribution format in BentoML. A Bento is packaged with all the source code, Python dependencies, model references, and environment setup, making it easy to deploy consistently across different environments.

Here is an example:

.. code-block:: python
    :caption: `service.py`

    my_image = bentoml.images.PythonImage(python_version="3.11") \
                .python_packages("mlflow", "scikit-learn")

    @bentoml.service(
        image=my_image, # Apply the specifications
        ...
    )
    class IrisClassifier:
        ...

Try it out
----------

You can run `this example project <https://github.com/bentoml/BentoMLflow>`_ on BentoCloud, or serve it locally, containerize it as an OCI-compliant image and deploy it anywhere.

.. _BentoCloud:

BentoCloud
^^^^^^^^^^

.. raw:: html

    <a id="bentocloud"></a>

BentoCloud provides fast and scalable infrastructure for building and scaling AI applications with BentoML in the cloud.

1. Install the dependencies and :doc:`log in to BentoCloud </scale-with-bentocloud/manage-api-tokens>` through the BentoML CLI. If you don't have a BentoCloud account, `sign up here for free <https://www.bentoml.com/>`_.

   .. code-block:: bash

      # Recommend Python 3.11
      pip install bentoml mlflow scikit-learn

      bentoml cloud login

2. Clone the repository.

   .. code-block:: bash

      git clone https://github.com/bentoml/BentoMLflow.git
      cd BentoMLflow

3. Train and save the MLflow model to the BentoML Model Store.

   .. code-block:: bash

      python3 save_model.py

4. Deploy the Service to BentoCloud.

   .. code-block:: bash

      bentoml deploy service.py:IrisClassifier

5. Once it is up and running, you can call the endpoint in the following ways:

   .. tab-set::

    .. tab-item:: BentoCloud Playground

		.. image:: ../../_static/img/examples/mlflow/mlflow-model-on-bentocloud.png
		   :alt: Screenshot of MLflow Iris classifier in the BentoCloud Playground interface showing how to interact with the deployed model

    .. tab-item:: Python client

       Create a :doc:`BentoML client </build-with-bentoml/clients>` to call the endpoint. Make sure you replace the Deployment URL with your own on BentoCloud. Refer to :ref:`scale-with-bentocloud/deployment/call-deployment-endpoints:obtain the endpoint url` for details.

       .. code-block:: python

          import bentoml

          with bentoml.SyncHTTPClient("https://iris-classifier-bdbe-e3c1c7db.mt-guc1.bentoml.ai") as client:
                result = client.predict(
                    input_data=[
                        [5.9, 3, 5.1, 1.8]
                    ],
                )
                print(result)

    .. tab-item:: CURL

       Make sure you replace the Deployment URL with your own on BentoCloud. Refer to :ref:`scale-with-bentocloud/deployment/call-deployment-endpoints:obtain the endpoint url` for details.

       .. code-block:: bash

          curl -X 'POST' \
                'https://iris-classifier-bdbe-e3c1c7db.mt-guc1.bentoml.ai/predict' \
                -H 'accept: application/json' \
                -H 'Content-Type: application/json' \
                -d '{
                "input_data": [
                    [5.9, 3, 5.1, 1.8]
                ]
            }'

6. To make sure the Deployment automatically scales within a certain replica range, add the scaling flags:

   .. code-block:: bash

      bentoml deploy --scaling-min 0 --scaling-max 3 # Set your desired count

   If it's already deployed, update its allowed replicas as follows:

   .. code-block:: bash

      bentoml deployment update <deployment-name> --scaling-min 0 --scaling-max 3 # Set your desired count

   For more information, see :doc:`how to configure concurrency and autoscaling </scale-with-bentocloud/scaling/autoscaling>`.

.. _LocalServing:

Local serving
^^^^^^^^^^^^^

.. raw:: html

    <a id="localserving"></a>

BentoML allows you to run and test your code locally, so that you can quickly validate your code with local compute resources.

1. Clone the project repository and install the dependencies.

   .. code-block:: bash

      git clone https://github.com/bentoml/BentoMLflow.git
      cd BentoMLflow

      # Recommend Python 3.11
      pip install bentoml mlflow scikit-learn

2. Train and save the model to the BentoML Model Store.

   .. code-block:: bash

      python3 save_model.py

3. Serve it locally.

   .. code-block:: bash

      bentoml serve service.py:IrisClassifier

4. Visit or send API requests to `http://localhost:3000 <http://localhost:3000/>`_.

For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
