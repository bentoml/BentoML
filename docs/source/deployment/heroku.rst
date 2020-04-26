Deploying to Heroku
===================

Heroku is a popular platform as a service based on managed container system. It provides
an easy solution to quickly build, run and scale applications.


Prerequsities
-------------

1. An active Heroku account and Heroku CLI tool installed in your system.

    * Install instruction: https://devcenter.heroku.com/articles/heroku-cli

2. Docker is installed and running on your system

    * Install instruction: https://docs.docker.com/install

3. Python 3.6 or above and required packages `bentoml` and `scikit-learn`:

    * .. code-block:: bash

            pip install bentoml scikit-learn



Heroku deployment with BentoML
------------------------------

This example builds a BentoService with iris classifier model, and deploy the
BentoService to Heroku as API server for inferencing.

Use the IrisClassifier BentoService from the getting started guide()

.. code-block:: bash

    git clone git@github.com:bentoml/BentoML.git
    python ./bentoml/guides/quick-start/main.py


.. code-block:: bash

    > bentoml get IrisClassifier:latest

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
            "handlerType": "DataframeHandler",
            "docs": "BentoService API"
          }
        ]
      }
    }


==========================
Build and deploy to Heroku
==========================


Make sure you have working Docker installation (eg. `docker ps`) and that you're logged
in to Heroku (`heroku login`)

Log in to the Heroku Container Registry:

.. code-block:: bash

    $ heroku container:login


Navigate to IrisClassifier SavedBundle directory:

.. code-block:: bash

    $ cd $(bentoml get IrisClassifier:latest -q | jq -r ".uri.uri")


Heroku requires HTTP traffic must be liston on `$PORT`, which is set by Heroku.  Update
the generated `Dockerfile` to meet this requirement. Better support for Heroku deployment
is coming soon.

Change the last line from `CMD ["bentoml serve-gunicorn /bento $FLAGS"]` to
`CMD bentoml serve-gunicorn /bento --port $PORT`.

Create Heroku app:

.. code-block:: bash

    $ heroku create

    #Sample output
    Creating app... done, ⬢ guarded-fjord-49167
    https://guarded-fjord-49167.herokuapp.com/ | https://git.heroku.com/guarded-fjord-49167.git

Build and push BentoService to your Heroku app:

.. code-block:: bash

    $ heroku container:push web --app APP_NAME


Release the app:

.. code-block:: bash

    $ heroku container:release web --app APP_NAME

Now, make prediction request with sample data:

.. code-block:: bash

    $ curl -i \
      --header "Content-Type: application/json" \
      --request POST \
      --data '[[5.1, 3.5, 1.4, 0.2]]' \
      ${heroku open --app APP_NAME}/predict


Remove deployment on Heroku

.. code-block:: bash

    $ heroku apps:destroy APP_NAME

