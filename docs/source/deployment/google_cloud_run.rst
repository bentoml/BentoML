Deploying to Google Cloud Run
=============================

Google Cloud Run is a fully manged compute platform that automatically scales. It is great
alternative to run BentoService that requires more computing power. Cloud Run is serverless. It
abstracts away infrastructure management, so you can focus on building service.

This guide demonstrates how to deploy a scikit-learn based iris classifier model with
BentoML to Google Cloud Run. The same deployment steps are also applicable for models
trained with other machine learning frameworks, see more BentoML examples :doc:`here <../examples>`.


Prerequisites
-------------

* Google cloud CLI tool

  * Install instruction: https://cloud.googl.com/sdk/install

* Docker is installed and running on the machine.

  * Install instruction: https://docs.docker.com/install

* Python 3.6 or above and required packages `bentoml` and `scikit-learn`:

  * .. code-block:: bash

        pip install bentoml scikit-learn

===========================
Create Google cloud project
===========================

.. code-block:: bash

    $ gcloud components update

    All components are up to date.


.. code-block:: bash

    $ gcloud projects create irisclassifier-gcloud-run

    # Sample output

    Create in progress for [https://cloudresourcemanager.googleapis.com/v1/projects/irisclassifier-gcloud-run].
    Waiting for [operations/cp.6403723248945195918] to finish...done.
    Enabling service [cloudapis.googleapis.com] on project [irisclassifier-gcloud-run]...
    Operation "operations/acf.15917ed1-662a-484b-b66a-03259041bf43" finished successfully.



.. code-block:: bash

    $ gcloud config set project irisclassifier-gcloud-run

    Updated property [core/project]


============================================================
Build and push BentoML model service image to GCP repository
============================================================

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


Use `gcloud` CLI to build the docker image

.. code-block:: bash

    # Install jq, the command-line JSON processor: https://stedolan.github.io/jq/download/
    $ saved_path=$(bentoml get IrisClassifier:latest -q | jq -r ".uri.uri")
    $ cd $saved_path
    $ gcloud builds submit --tag gcr.io/irisclassifier-gcloud-run/iris-classifier

    # Sample output

    Creating temporary tarball archive of 15 file(s) totalling 15.8 MiB before compression.
    Uploading tarball of [.] to [gs://irisclassifier-gcloud-run_cloudbuild/source/1587430763.39-03422068242448efbcfc45f2aed218d3.tgz]
    Created [https://cloudbuild.googleapis.com/v1/projects/irisclassifier-gcloud-run/builds/9c0f3ef4-11c0-4089-9406-1c7fb9c7e8e8].
    Logs are available at [https://console.cloud.google.com/cloud-build/builds/9c0f3ef4-11c0-4089-9406-1c7fb9c7e8e8?project=349498001835]
    ----------------------------- REMOTE BUILD OUTPUT ------------------------------
    ...
    ...
    ...
    DONE
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ID                                    CREATE_TIME                DURATION  SOURCE                                                                                               IMAGES                                                      STATUS
    9c0f3ef4-11c0-4089-9406-1c7fb9c7e8e8  2020-04-21T00:59:38+00:00  5M22S     gs://irisclassifier-gcloud-run_cloudbuild/source/1587430763.39-03422068242448efbcfc45f2aed218d3.tgz  gcr.io/irisclassifier-gcloud-run/iris-classifier (+1 more)  SUCCESS


====================================
Deploy the image to Google Cloud Run
====================================

1. Use your browser, go into the Google Cloud Console, select project `sentiment-gcloud-run` and navigate to the CloudRun page
2. Click `Create Service` on the top of the navigation bar
3. In the Create Cloud Run service page:

**Select container image URL from the selection menu, choose allow Unauthenticated invocations from the Authentication section**

.. image:: ../_static/img/gcloud-start.png
    :alt: GCP project creation

**Expand Show Optional Revision Settings and change Container Port from `8080` to `5000`**

.. image:: ../_static/img/gcloud-setting.png
    :alt: GCP project setting

After successful deployment, you can fin the service endpoint URL at top of the page.

.. image:: ../_static/img/gcloud-endpoint.png
    :alt: GCP project endpoint


=====================================================
Validate Google cloud run deployment with sample data
=====================================================

Copy the service URL from the screen

.. code-block:: bash

    $ curl -i \
        --header "Content-Type: application/json" \
        --request POST \
        -d '[[5.1, 3.5, 1.4, 0.2]]' \
        https://iris-classifier-7v6yobzlcq-uw.a.run.app/predict

    # Sample output
    [0]


=============================================
Clean up deployed service on Google Cloud Run
=============================================

1. Navigate to the manage resources page in Google Cloud Console.
2. In the project list, select the project you want to delete and click the `delete` icon
3. In the dialog, type the projectID `sentiment-gcloud-run` and then click `Shut down` to delete the project.
