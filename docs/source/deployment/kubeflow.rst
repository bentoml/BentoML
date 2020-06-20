Deploying to Kubeflow
=====================

Kubeflow is the machine learning toolkit for Kubernetes. It makes producing ML services
as simple as possible from data preparation to service management.

This guide demonstrates how to serve a scikit-learn based iris classifier model with
BentoML on a Kubernetes cluster. The same deployment steps are also applicable for models
trained with other machine learning frameworks, see more BentoML examples :doc:`here <../examples>`.


==============
Prerequisites
==============

Before starting this guide, make sure you have the following:

* a Kubernetes cluster and kubectl installed on your local machine.

  * kubectl install instruction: https://kubernetes.io/docs/tasks/tools/install-kubectl/

* Docker and Docker Hub installed and configured in your local machine.

  * Docker install instruction: https://docs.docker.com/get-docker/

* Python 3.6 or above and required PyPi packages: `bentoml` and `scikit-learn`

  * .. code-block:: bash

          pip install bentoml scikit-learn


Kubeflow deployment with BentoML
--------------------------------

Run the example project from the :doc:`quick start guide <../quickstart>` to create the
BentoML saved bundle for deployment:

.. code-block:: bash

    git clone git@github.com:bentoml/BentoML.git
    python ./bentoml/guides/quick-start/main.py


Use BentoML CLI tool to get the information of IrisClassifier created above

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


===============================
Deploy BentoService to Kubeflow
===============================

BentoML provides a convenient way to containerize the model API server with Docker:

    1. Find the SavedBundle directory with `bentoml get` command

    2. Run docker build with the SavedBundle directory which contains a generated Dockerfile

    3. Run the generated docker image to start a docker container serving the model

.. code-block:: bash

    # Install jq, the command-line JSON processor: https://stedolan.github.io/jq/download/
    saved_path=$(bentoml get IrisClassifier:latest -q | jq -r ".uri.uri")

    # Replace {docker_username} with your Docker Hub username
    docker build -t {docker_username}/iris-classifier $saved_path
    docker push {docker_username}/iris-classifier


The following is an example YAML file for specifying the resources required to run and
expose a BentoML model server in a Kubernetes cluster. Replace {docker_username} with
your Docker Hub username and save it to iris-classifier.yaml

.. code-block:: yaml

    #iris-classifier.yaml

    apiVersion: v1
    kind: Service
    metadata:
        labels:
            app: iris-classifier
        name: iris-classifier
        namespace: kubeflow
    spec:
        ports:
        - name: predict
          port: 5000
          targetPort: 5000
        selector:
          app: iris-classifier
        type: LoadBalancer
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
        labels:
            app: iris-classifier
        name: iris-classifier
        namespace: kubeflow
    spec:
        selector:
            matchLabels:
                app: iris-classifier
        template:
            metadata:
                labels:
                    app: iris-classifier
            spec:
                containers:
                - image: {docker_username}/iris-classifier
                  imagePullPolicy: IfNotPresent
                  name: iris-classifier
                  ports:
                  - containerPort: 5000


Use `kubectl` CLI to deploy model server to the cluster.

.. code-block:: bash

    kubectl apply -f iris-classifier.yaml


Make prediction with `curl`:

.. code-block:: bash

    curl -i \
    --request POST \
    --header "Content-Type: application/json" \
    --data '[[5.1, 3.5, 1.4, 0.2]]' \
    ${minikube ip}:5000/predict


============================================
Monitor model server metrics with Prometheus
============================================


Setup:

Before starting this section, make sure you have the following:

* Prometheus installed on your Kubernetes cluster

  * installation instruction: https://github.com/coreos/kube-prometheus


BentoML API server provides Prometheus support out of the box. It comes with a “/metrics”
endpoint which includes the essential metrics for model serving and the ability to create
and customize new metrics base on needs.

To enable Prometheus monitoring on the deployed model API server, update the YAML file
with Prometheus related annotations. Change the deployment spec as the following, and
replace `{docker_username}` with your Docker Hub username:


.. code-block:: bash

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      labels:
        app: pet-classifier
      name: pet-classifier
      namespace: kubeflow
    spec:
      selector:
        matchLabels:
          app: pet-classifier
      template:
        metadata:
          labels:
            app: pet-classifier
          annotations:
            prometheus.io/scrape: true
            prometheus.io/port: 5000
        spec:
          containers:
          - image: {docker_username}/pet-classifier
            name: pet-classifier
            ports:
            - containerPort: 5000


Apply the changes to enable monitoring:

.. code-block:: bash

    kubectl apply -f iris-classifier.yaml



=================
Remove deployment
=================

.. code-block:: bash

    kubectl delete -f iris-classifier.yaml

