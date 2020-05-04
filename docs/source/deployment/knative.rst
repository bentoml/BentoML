Deploying to Knative
====================

Knative is kubernetes based platform to deploy and manage serverless workloads. It is a
solution for deploying ML workload that requires more computing power that abstracts away
infrastructure management and without worry about vendor lock.


Prerequisites
-------------

* kubernetes cluster version 1.15 or newer.

* Knative with istio as network layer

    * Knative install instruction: https://knative.dev/docs/install/any-kubernetes-cluster/

    * Install istio for knative: https://knative.dev/docs/install/installing-istio/

* Saved BentoService bundle

    * for this guide, we are using the IrisClassifier that was created in the
      quick start guide: https://docs.bentoml.org/en/latest/quickstart.html


Knative deployment with BentoML
-------------------------------

In this guide, we will build BentoService with docker and then deploy a
prediction services on Knative with the docker image.

===========================
Build and push docker image
===========================

First, we need to build and push our BentoService to a docker registry.
We will use the IrisClassifier BentoService from the getting
:doc:`Quick start guide<../quickstart>`.

.. code-block:: bash

    git clone git@github.com:bentoml/BentoML.git
    python ./bentoml/guides/quick-start/main.py

Use BentoML CLI tool to get the information about IrisClassifier.

.. code-block:: bash

    > bentoml get IrisClassifier:20200121141808_FE78B5

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


Navigate to the BentoService archive bundle location. Build and push docker image to docker hub.


.. code-block:: bash

    > cd /Users/bozhaoyu/bentoml/repository/IrisClassifier/20200121141808_FE78B
    > docker build . -t yubozhao/iris-classifier
    > docker push yubozhao/iris-classifier


=================
Deploy to Knative
=================


Make sure Knative serving component and its pods are running.

.. code-block:: bash

    > kubectl get pods --namespace knative-serving
    NAME                                READY   STATUS    RESTARTS   AGE
    activator-845b77cbb5-thpcw          2/2     Running   0          4h33m
    autoscaler-7fc56894f5-f2vqc         2/2     Running   0          4h33m
    controller-7ffb84fd9c-699pt         2/2     Running   2          4h33m
    networking-istio-7fc7f66675-xgfvd   1/1     Running   0          4h32m
    webhook-8597865965-9vp25            2/2     Running   1          4h33m


Create a service.yaml file and copy the following service definition into the file. We are pointing
livenessProbe and readyinessProbe to the /healthz endpoint on BentoService.



.. code-block:: yaml

    apiVersion: serving.knative.dev/v1
    kind: Service
    metadata:
      name: iris-classifier
      namespace: bentoml
    spec:
      template:
        spec:
          containers:
            - image: docker.io/yubozhao/iris-classifier
              ports:
              - containerPort: 5000
              livenessProbe:
                httpGet:
                  path: /healthz
                initialDelaySeconds: 3
                periodSeconds: 5
              readinessProbe:
                httpGet:
                  path: /healthz
                initialDelaySeconds: 3
                periodSeconds: 5
                failureThreshold: 3
                timeoutSeconds: 60



Create bentoml namespace and then deploy BentoService to Knative with kubectl apply command.

.. code-block:: bash

    > kubectl create namespace bentoml
    > kubectl apply -f service.yaml
    service.serving.knative.dev/iris-classifier created



We can monitor the status with kubectl get ksvc command.

.. code-block:: bash

    > kubectl get ksvc --all-namespaces
    NAMESPACE   NAME              URL                                          LATESTCREATED           LATESTREADY             READY   REASON
    bentoml     iris-classifier   http://iris-classifier.bentoml.example.com   iris-classifier-7k2dv   iris-classifier-7k2dv   True


===========================================
Validate prediction server with sample data
===========================================


For this guide, our kubernetes cluster run on minikube, we will get the appropriate ip from minikube and the port from istio

.. code-block::

    > minikube ip
    192.168.64.4

    > kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].nodePort}
    31871


With the ip address and port, we can make a curl request to the prediction result from Knative

.. code-block:: bash

    > curl -v -i \
        --header "Content-Type: application/json" \
        --header "Host: iris-classifier.bentoml.example.com" \
        --request POST \
        --data '[[5.1, 3.5, 1.4, 0.2]]' \
        http://192.168.64.4:31871/predict

    Note: Unnecessary use of -X or --request, POST is already inferred.
    *   Trying 192.168.64.4...
    * TCP_NODELAY set
    * Connected to 192.168.64.4 (192.168.64.4) port 31871 (#0)
    > POST /predict HTTP/1.1
    > Host: iris-classifier.bentoml.example.com
    > User-Agent: curl/7.58.0
    > Accept: */*
    > Content-Type: application/json
    > Content-Length: 22
    >
    * upload completely sent off: 22 out of 22 bytes
    < HTTP/1.1 200 OK
    HTTP/1.1 200 OK
    < content-length: 3
    content-length: 3
    < content-type: application/json
    content-type: application/json
    < date: Wed, 01 Apr 2020 01:24:58 GMT
    date: Wed, 01 Apr 2020 01:24:58 GMT
    < request_id: 0506467b-75d9-4fb5-9d7e-2d2855fc6028
    request_id: 0506467b-75d9-4fb5-9d7e-2d2855fc6028
    < server: istio-envoy
    server: istio-envoy
    < x-envoy-upstream-service-time: 12
    x-envoy-upstream-service-time: 12

    <
    * Connection #0 to host 192.168.64.4 left intact
    [0]%


===================
Clean up deployment
===================

.. code-block:: bash

    > kubectl delete namespace bentoml
