Deploying to Heroku
===================

Heroku is a popular platform as a service based on managed container system. It provides
an easy solution to quickly build, run and scale applications. BentoML works great with
Heroku. BentoServices could quickly deploy to Heroku as API model server for production.

This guide demonstrates how to deploy a scikit-learn based iris classifier model with
BentoML to Heroku. The same deployment steps are also applicable for models
trained with other machine learning frameworks, see more BentoML examples :doc:`here <../examples>`.


Prerequisites
-------------

* An active Heroku account and Heroku CLI tool installed in your system.

    * Install instruction: https://devcenter.heroku.com/articles/heroku-cli

* Docker is installed and docker daemon is running on your system

    * Install instruction: https://docs.docker.com/install

* Python 3.6 or above and required packages `bentoml` and `scikit-learn`:

    * .. code-block:: bash

            pip install bentoml scikit-learn



Heroku deployment with BentoML
------------------------------

Run the example project from the :doc:`quick start guide <../quickstart>` to create the
BentoML saved bundle for deployment:


.. code-block:: bash

    git clone git@github.com:bentoml/BentoML.git
    pip install -r ./bentoml/guides/quick-start/requirements.txt
    python ./bentoml/guides/quick-start/main.py

Verify the saved bundle created:

.. code-block:: bash

    $ bentoml get IrisClassifier:latest

    # Sample output

    {
      "name": "IrisClassifier",
      "version": "20200121141808_FE78B5",
      "uri": {
        "type": "LOCAL",
        "uri": "/Users/bozhaoyu/bentoml/repository/IrisClassifier/20200121141808_FE78B5"
      },
      "bentoServiceMetadata": {
        "name": "IrisClassifier",
        "version": "20200121141808_FE78B5",
        "createdAt": "2020-01-21T22:18:25.079723Z",
        "env": {
          "condaEnv": "name: bentoml-IrisClassifier\nchannels:\n- defaults\ndependencies:\n- python=3.7.3\n- pip\n",
          "pipDependencies": "bentoml==0.5.8\nscikit-learn",
          "pythonVersion": "3.7.3"
        },
        "artifacts": [
          {
            "name": "model",
            "artifactType": "SklearnModelArtifact"
          }
        ],
        "apis": [
          {
            "name": "predict",
            "InputType": "DataframeInput",
            "docs": "BentoService API"
          }
        ]
      }
    }


The BentoML saved bundle created can now be used to start a REST API Server hosting the
BentoService and available for sending test request:

.. code-block:: bash

    # Start BentoML API server:
    bentoml serve IrisClassifier:latest


.. code-block:: bash

    # Send test request:
    curl -i \
      --header "Content-Type: application/json" \
      --request POST \
      --data '[[5.1, 3.5, 1.4, 0.2]]' \
      http://localhost:5000/predict


==========================
Build and deploy to Heroku
==========================

1. Download and Install BentoML Heroku deployment tool

.. code-block:: bash

    git clone https://github.com/bentoml/heroku-deploy.git
    cd heroku-deploy
    pip install -r requirements.txt


2. Create a Heroku deloyment

.. code-block:: bash

    BENTO_BUNDLE=$(bentoml get IrisClassifier:latest --print-location -q)
    python deploy.py $BENTO_BUNDLE my_deployment heroku_config.json

3. Get deployment information

.. code-block:: bash

    python describe.py my_deployment

4. Make request to Heroku deployment

.. code-block:: bash

    curl  curl -i \
    --header "Content-Type: application/json" \
    --request POST \
    --data '[[5.1, 3.5, 1.4, 0.2]]' \
    https://btml-my_deployment.herokuapp.com/predict

5. Delete Heroku deployment

.. code-block:: bash

    python delete.py my_deployment



