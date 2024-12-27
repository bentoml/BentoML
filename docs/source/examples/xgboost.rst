=======
XGBoost
=======

`XGBoost <https://xgboost.readthedocs.io/en/stable/>`_ is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework.

This document explains how to serve and deploy an XGBoost model for predicting breast cancer with BentoML. You can find all the source code `here <https://github.com/bentoml/BentoXGBoost>`_.

Prerequisites
-------------

- Python 3.9+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read :doc:`/get-started/hello-world` first.
- (Optional) We recommend you create a virtual environment for dependency isolation. See the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or the `Python documentation <https://docs.python.org/3/library/venv.html>`_ for details.

Install dependencies
--------------------

.. code-block:: bash

    pip install xgboost bentoml scikit-learn

Train and save a model
----------------------

This example uses the ``scikit-learn`` framework to load and preprocess the `breast cancer dataset <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>`_, which is then converted into an XGBoost-compatible format (``DMatrix``) to train the machine learning model.

.. code-block:: python

    import typing as t
    from sklearn.datasets import load_breast_cancer
    from sklearn.utils import Bunch
    import xgboost as xgb

    # Load the data
    cancer: Bunch = t.cast("Bunch", load_breast_cancer())
    cancer_data = t.cast("ext.NpNDArray", cancer.data)
    cancer_target = t.cast("ext.NpNDArray", cancer.target)
    dt = xgb.DMatrix(cancer_data, label=cancer_target)

    # Specify model parameters
    param = {
        "max_depth": 3,
        "eta": 0.3,
        "objective": "multi:softprob",
        "num_class": 2
    }

    # Train the model
    model = xgb.train(param, dt)

After training, use the ``bentoml.xgboost.save_model`` API to save the model to the BentoML :doc:`Model Store </build-with-bentoml/model-loading-and-management>`, a local directory to store and manage models. You can retrieve this model later in other services to run predictions.

.. code-block:: bash

    import bentoml

    # Specify the model name and the model to be saved
    bentoml.xgboost.save_model("cancer", model)

To verify that the model has been successfully saved, run:

.. code-block:: bash

    $ bentoml models list

    Tag                      Module           Size       Creation Time
    cancer:xa2npbboccvv7u4c  bentoml.xgboost  23.17 KiB  2024-06-19 07:51:21

Test the saved model
--------------------

To ensure that the saved model works correctly, try loading it and running a prediction:

.. code-block:: python

    import bentoml
    import xgboost as xgb

    # Load the model by setting the model tag
    booster = bentoml.xgboost.load_model("cancer:xa2npbboccvv7u4c")

    # Predict using a sample
    res = booster.predict(xgb.DMatrix([[1.308e+01, 1.571e+01, 8.563e+01, 5.200e+02, 1.075e-01, 1.270e-01,
        4.568e-02, 3.110e-02, 1.967e-01, 6.811e-02, 1.852e-01, 7.477e-01,
        1.383e+00, 1.467e+01, 4.097e-03, 1.898e-02, 1.698e-02, 6.490e-03,
        1.678e-02, 2.425e-03, 1.450e+01, 2.049e+01, 9.609e+01, 6.305e+02,
        1.312e-01, 2.776e-01, 1.890e-01, 7.283e-02, 3.184e-01, 8.183e-02]]))

    print(res)

Expected result:

.. code-block:: bash

    [[0.02664177 0.9733583 ]] # The probability of the sample belonging to class 0 and class 1

Create a BentoML Service
------------------------

Create a separate ``service.py`` file where you define a BentoML :doc:`Service </build-with-bentoml/services>` to expose the model as a web service.

.. code-block:: python

    import bentoml
    import numpy as np
    import xgboost as xgb
    import os

    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
    )
    class CancerClassifier:
        # Retrieve the latest version of the model from the BentoML Model Store
        bento_model = bentoml.models.get("cancer:latest")

        def __init__(self):
            self.model = bentoml.xgboost.load_model(self.bento_model)

            # Check resource availability
            if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
                self.model.set_param({"predictor": "gpu_predictor", "gpu_id": 0})  # type: ignore (incomplete XGBoost types)
            else:
                nthreads = os.getenv("OMP_NUM_THREADS")
                if nthreads:
                    nthreads = max(int(nthreads), 1)
                else:
                    nthreads = 1
                self.model.set_param(
                    {"predictor": "cpu_predictor", "nthread": nthreads}
                )

        @bentoml.api
        def predict(self, data: np.ndarray) -> np.ndarray:
            return self.model.predict(xgb.DMatrix(data))

The Service code:

- Uses the ``@bentoml.service`` decorator to define a BentoML Service. Optionally, you can set additional configurations like resource allocation and traffic timeout.
- Retrieves the model from the Model Store and defines it a class variable.
- Checks resource availability like GPUs and the number of threads.
- Uses the ``@bentoml.api`` decorator to expose the ``predict`` function as an API endpoint, which :doc:`takes a NumPy array as input and returns a NumPy array </build-with-bentoml/iotypes>`. Note that the input data is converted into a ``DMatrix``, which is the data structure XGBoost uses for datasets.

Run ``bentoml serve`` in your project directory to start the Service.

.. code-block:: bash

    $ bentoml serve service:CancerClassifier

    2024-06-19T08:37:31+0000 [WARNING] [cli] Converting 'CancerClassifier' to lowercase: 'cancerclassifier'.
    2024-06-19T08:37:31+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:CancerClassifier" listening on http://localhost:3000 (Press CTRL+C to quit)

The server is active at `http://localhost:3000 <http://localhost:3000/>`_. You can interact with it in different ways.

.. tab-set::

    .. tab-item:: CURL

        .. code-block:: bash

            curl -X 'POST' \
                'http://localhost:3000/predict' \
                -H 'accept: application/json' \
                -H 'Content-Type: application/json' \
                -d '{
                "data": [
                    [1.308e+01, 1.571e+01, 8.563e+01, 5.200e+02, 1.075e-01, 1.270e-01,
                    4.568e-02, 3.110e-02, 1.967e-01, 6.811e-02, 1.852e-01, 7.477e-01,
                    1.383e+00, 1.467e+01, 4.097e-03, 1.898e-02, 1.698e-02, 6.490e-03,
                    1.678e-02, 2.425e-03, 1.450e+01, 2.049e+01, 9.609e+01, 6.305e+02,
                    1.312e-01, 2.776e-01, 1.890e-01, 7.283e-02, 3.184e-01, 8.183e-02]
                  ]
                }'

    .. tab-item:: Python client

        .. code-block:: python

            import bentoml

            with bentoml.SyncHTTPClient("http://localhost:3000") as client:
                result = client.predict(
                    data=[
                        [1.308e+01, 1.571e+01, 8.563e+01, 5.200e+02, 1.075e-01, 1.270e-01,
                        4.568e-02, 3.110e-02, 1.967e-01, 6.811e-02, 1.852e-01, 7.477e-01,
                        1.383e+00, 1.467e+01, 4.097e-03, 1.898e-02, 1.698e-02, 6.490e-03,
                        1.678e-02, 2.425e-03, 1.450e+01, 2.049e+01, 9.609e+01, 6.305e+02,
                        1.312e-01, 2.776e-01, 1.890e-01, 7.283e-02, 3.184e-01, 8.183e-02]
                    ],
                )
                print(result)

    .. tab-item:: Swagger UI

        Visit `http://localhost:3000 <http://localhost:3000/>`_, scroll down to **Service APIs**, specify the data, and click **Execute**.

        .. image:: ../../_static/img/examples/xgboost/service-ui.png

Deploy to BentoCloud
--------------------

After the Service is ready, you can deploy it to BentoCloud for better management and scalability. `Sign up <https://www.bentoml.com/>`_ for a BentoCloud account and get $10 in free credits.

First, specify a configuration YAML file (``bentofile.yaml``) to define the build options for a :doc:`Bento </reference/bentoml/bento-build-options>`, the unified distribution format in BentoML containing source code, Python packages, model references, and so on. Here is an example file:

.. code-block:: yaml

    service: "service:CancerClassifier"
    labels:
      owner: bentoml-team
      stage: demo
    include:
      - "*.py"
    python:
      packages:
        - xgboost
        - scikit-learn

:ref:`Log in to BentoCloud <scale-with-bentocloud/manage-api-tokens:Log in to BentoCloud using the BentoML CLI>` by running ``bentoml cloud login``, then run the following command to deploy the project.

.. code-block:: bash

    bentoml deploy .

Once the Deployment is up and running on BentoCloud, you can access it via the exposed URL.

.. image:: ../../_static/img/examples/xgboost/bentocloud-ui.png

.. note::

   For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
