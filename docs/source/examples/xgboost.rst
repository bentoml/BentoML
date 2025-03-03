=======
XGBoost
=======

`XGBoost <https://xgboost.readthedocs.io/en/stable/>`_ is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework.

This document explains how to serve and deploy an XGBoost model for predicting breast cancer with BentoML.

.. raw:: html

    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; flex-grow: 1; margin-right: 10px; text-align: center;">
            <img src="https://docs.bentoml.com/en/latest/_static/img/github-mark.png" alt="GitHub" style="vertical-align: middle; width: 24px; height: 24px;">
            <a href="https://github.com/bentoml/BentoXGBoost" style="margin-left: 5px; vertical-align: middle;">Source Code</a>
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

You can query the exposed prediction endpoint with breast tumor data. For example:

.. code-block:: bash

   {
        "data": [
                    [1.308e+01, 1.571e+01, 8.563e+01, 5.200e+02, 1.075e-01, 1.270e-01,
                    4.568e-02, 3.110e-02, 1.967e-01, 6.811e-02, 1.852e-01, 7.477e-01,
                    1.383e+00, 1.467e+01, 4.097e-03, 1.898e-02, 1.698e-02, 6.490e-03,
                    1.678e-02, 2.425e-03, 1.450e+01, 2.049e+01, 9.609e+01, 6.305e+02,
                    1.312e-01, 2.776e-01, 1.890e-01, 7.283e-02, 3.184e-01, 8.183e-02]
                ]
    }

Expected output:

.. code-block:: bash

   [[0.02664177 0.9733583 ]] # 2.66% chance benign and 97.34% chance malignant

This example is ready for quick deployment and scaling on BentoCloud. With a single command, you get a production-grade application with fast autoscaling, secure deployment in your cloud, and comprehensive observability.

.. image:: ../../_static/img/examples/xgboost/xgboost-model-running-on-bentocloud.png
    :alt: Screenshot of the XGBoost breast cancer classifier model running on BentoCloud showing the prediction interface with input data and results

Code explanations
-----------------

You can find `the source code in GitHub <https://github.com/bentoml/BentoXGBoost>`_. Below is a breakdown of the key code implementations within this project.

save_model.py
^^^^^^^^^^^^^

This example uses the ``scikit-learn`` framework to load and preprocess the `breast cancer dataset <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>`_, which is then converted into an XGBoost-compatible format (``DMatrix``) to train the machine learning model.

.. code-block:: python
    :caption: `save_model.py`

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

test.py
^^^^^^^

To ensure that the saved model works correctly, try loading it and running a prediction:

.. code-block:: python
    :caption: `test.py`

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

    [[0.02664177 0.9733583 ]]

service.py
^^^^^^^^^^

The ``service.py`` file is where you define the serving logic and expose the model as a web service.

.. code-block:: python
    :caption: `service.py`

    import bentoml
    import numpy as np
    import xgboost as xgb
    import os

    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
    )
    class CancerClassifier:
        # Declare the model as a class variable
        bento_model = bentoml.models.BentoModel("cancer:latest")

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

- Uses the ``@bentoml.service`` decorator to define a BentoML :doc:`Service </build-with-bentoml/services>`. Optionally, you can set additional :doc:`configurations </reference/bentoml/configurations>` like resource allocation on BentoCloud and traffic timeout.
- Retrieves the model from the Model Store and defines it a class variable.
- Checks resource availability like GPUs and the number of threads.
- Uses the ``@bentoml.api`` decorator to expose the ``predict`` function as an API endpoint, which :doc:`takes a NumPy array as input and returns a NumPy array </build-with-bentoml/iotypes>`. Note that the input data is converted into a ``DMatrix``, which is the data structure XGBoost uses for datasets.

The ``@bentoml.service`` decorator also allows you to :doc:`define the runtime environment </build-with-bentoml/runtime-environment>` for a Bento, the unified distribution format in BentoML. A Bento is packaged with all the source code, Python dependencies, model references, and environment setup, making it easy to deploy consistently across different environments.

Here is an example:

.. code-block:: python
    :caption: `service.py`

    my_image = bentoml.images.PythonImage(python_version="3.11") \
                .python_packages("xgboost", "scikit-learn")

    @bentoml.service(
        image=my_image, # Apply the specifications
        ...
    )
    class CancerClassifier:
        ...

Try it out
----------

You can run `this example project <https://github.com/bentoml/BentoXGBoost>`_ on BentoCloud, or serve it locally, containerize it as an OCI-compliant image and deploy it anywhere.

.. _BentoCloud:

BentoCloud
^^^^^^^^^^

.. raw:: html

    <a id="bentocloud"></a>

BentoCloud provides fast and scalable infrastructure for building and scaling AI applications with BentoML in the cloud.

1. Install the dependencies and :doc:`log in to BentoCloud </scale-with-bentocloud/manage-api-tokens>` through the BentoML CLI. If you don't have a BentoCloud account, `sign up here for free <https://www.bentoml.com/>`_.

   .. code-block:: bash

      # Recommend Python 3.11
      pip install bentoml xgboost scikit-learn

      bentoml cloud login

2. Clone the repository.

   .. code-block:: bash

      git clone https://github.com/bentoml/BentoXGBoost.git
      cd BentoXGBoost

3. Train and save the MLflow model to the BentoML Model Store.

   .. code-block:: bash

      python3 save_model.py

4. Deploy the Service to BentoCloud.

   .. code-block:: bash

      bentoml deploy

5. Once it is up and running, you can call the endpoint in the following ways:

   .. tab-set::

    .. tab-item:: BentoCloud Playground

	   .. image:: ../../_static/img/examples/xgboost/xgboost-model-running-on-bentocloud.png
	      :alt: Screenshot of XGBoost breast cancer classifier in the BentoCloud Playground interface showing how to interact with the deployed model

    .. tab-item:: Python client

       Create a :doc:`BentoML client </build-with-bentoml/clients>` to call the endpoint. Make sure you replace the Deployment URL with your own on BentoCloud. Refer to :ref:`scale-with-bentocloud/deployment/call-deployment-endpoints:obtain the endpoint url` for details.

       .. code-block:: python

          import bentoml

          with bentoml.SyncHTTPClient("https://cancer-classifier-33e8-e3c1c7db.mt-guc1.bentoml.ai") as client:
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

    .. tab-item:: CURL

       Make sure you replace the Deployment URL with your own on BentoCloud. Refer to :ref:`scale-with-bentocloud/deployment/call-deployment-endpoints:obtain the endpoint url` for details.

       .. code-block:: bash

          curl -X 'POST' \
                'https://cancer-classifier-33e8-e3c1c7db.mt-guc1.bentoml.ai/predict' \
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

      git clone https://github.com/bentoml/BentoXGBoost.git
      cd BentoXGBoost

      # Recommend Python 3.11
      pip install bentoml xgboost scikit-learn

2. Train and save the model to the BentoML Model Store.

   .. code-block:: bash

      python3 save_model.py

3. Serve it locally.

   .. code-block:: bash

      bentoml serve

4. Visit or send API requests to `http://localhost:3000 <http://localhost:3000/>`_.

For custom deployment in your own infrastructure, use BentoML to :doc:`generate an OCI-compliant image </get-started/packaging-for-deployment>`.
