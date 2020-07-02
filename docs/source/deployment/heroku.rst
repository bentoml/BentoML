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


Follow the CLI instruction and login to a Heroku account:

.. code-block:: bash

    heroku login

Login to the Heroku Container Registry:

.. code-block:: bash

    heroku container:login


Create a Heroku app:

.. code-block:: bash

    APP_NAME=bentoml-her0ku-$(date +%s | base64 | tr '[:upper:]' '[:lower:]' | tr -dc _a-z-0-9)
    heroku create $APP_NAME


Find the IrisClassifier SavedBundle directory:

.. code-block:: bash

    # Install jq, the command-line JSON processor: https://stedolan.github.io/jq/download/
    cd $(bentoml get IrisClassifier:latest -q | jq -r ".uri.uri")


Build and push API server container with the SavedBundle, and push to the Heroku app
`bentoml-iris-classifier` created above:

.. code-block:: bash

    heroku container:push web --app $APP_NAME


Release the app:

.. code-block:: bash

    heroku container:release web --app $APP_NAME


To view the deployment logs on heroku and verify the web server has been created:

.. code-block:: bash

    heroku logs --tail -a $APP_NAME

Now, make prediction request with sample data:

.. code-block:: bash

    curl -i \
      --header "Content-Type: application/json" \
      --request POST \
      --data '[[5.1, 3.5, 1.4, 0.2]]' \
      $(heroku apps:info --app $APP_NAME -j | jq -r ".app.web_url")/predict


Remove deployment on Heroku

.. code-block:: bash

    heroku apps:destroy $APP_NAME

