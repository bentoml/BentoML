Deploying to KFServing
======================


This guide demostrates how to deploy a BentoService to a KFServing cluster.

KFServing enables serverless inferencing on Kubernetes cluster for common machine learning
frameworks like Tensorflow, XGBoost, scikit-learn and etc. BentoServices can easily
deploy to KFServing and take advantage of what KFServing offers.


=============
Prerequisites
=============

Before starting this guide, make sure you have the following:

* a cluster with KFServing installed

* Docker and Docker Hub installed and configured on your local machine.

  * Docker install instruction: https://docs.docker.com/get-docker/

* Python 3.6 or above and required PyPi packages: `bentoml` and `scikit-learn`

  * .. code-block: bash

          pip install bentoml scikit-learn


KFServing deployment with BentoML
---------------------------------

This guide use the IrisClassifier BentoService from the quick start guide:

.. code-block:: bash

    git clone git@github.com:bentoml/BentoML.git
    python ./bentoml/guides/quick-start/main.py

Use BentoML CLI tool to get the information of IrisClassifier created above:

.. code-block:: bash

    bentoml get IrisClassifier:latest

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

After saving the BentoService instance, you can now start a REST API server with the
model trained and test the API server locally:

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

================================
Deploy BentoService to KFServing
================================

BentoML provides a convenient way of containerizing the model API server with Docker. To
create a docker container image for the sample model above:

  1. Find the file directory of the SavedBundle with `bentoml get` command, which is
  directory structured as a docker build context.

  2. Running docker build with this directory produces a docker image containing the API
  model server.

.. code-block:: bash

    model_path=$(bentoml get IrisClassifier:latest -q | jq -r ".uri.uri")

    # Replace {docker_username} with your Docker Hub username
    docker build -t {docker_username}/iris-classifier $model_path
    docker push {docker_username}/iris-classifier


*Note: BentoML's REST interface is different than the Tensorflow V1 HTTP API that
KFServing expects. Requests will send directly to the prediction service and bypass the
top-level InferenceService.*

*Support for KFServing V2 prediction protocol with BentoML is coming soon.*

The following is an example YAML file for specifying the resources required to run an
InferenceService in KFServing. Replace `{docker_username}` with your Docker Hub username
and save it to `bentoml.yaml` file:

.. code-block:: yaml

    apiVersion: serving.kubeflow.org/v1alpha2
    kind: InferenceService
    metadata:
      labels:
        controller-tools.k8s.io: "1.0"
      name: iris-classifier
    spec:
      default:
        predictor:
          custom:
            container:
              image: {docker_username}/iris-classifier
              ports:
                - containerPort: 5000

Use `kubectl apply` command to deploy the InferenceService:

.. code-block:: bash

    kubectl apply -f bentoml.yaml

==============
Run prediction
==============

*Note: Use kfserving-ingressgateway as your INGRESS_GATEWAY if you are deploying
KFServing as part of Kubeflow install, and not independently.*

.. code-block:: bash

    MODEL_NAME=iris-classifier
    INGRESS_GATEWAY=istio-ingressgateway
    CLUSTER_IP=$(kubectl -n istio-system get service $INGRESS_GATEWAY -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    SERVICE_HOSTNAME=$(kubectl get inferenceservice ${MODEL_NAME} -o jsonpath='{.status.url}' | cut -d "/" -f 3)

    curl -v -H "Host: ${SERVICE_HOSTNAME}" \
      --header "Content-Type: application/json" \
      --request POST \
      --data '[[5.1, 3.5, 1.4, 0.2]]' \
      http://$CLUSTER_IP/model/predict

### Delete deployment

.. code-block:: bash

    kubectl delete -f bentoml.yaml