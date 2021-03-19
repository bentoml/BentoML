==============================
Install Yatai on K8w with Helm
==============================

*Helm* is a tool that streamlines installing and managing K8s applications. It lets developers easily configure a set of K8s resources that make up an application -- YataiService in this case. Helm enables developers to

- easily deploy recommended configuration and setup for YataiService
- easily change settings
- drop and replace parts with your own (e.g. replace local Postgres with a remote RDS)

1. Configuration
================
Helm charts rely on a values file to configure how to template and create Kubernetes resources. The default values for these resources can be found in `BentoML/helm/YataiService/values.yaml`. These basic values describe a basic local `YataiService` instance with ephemeral storage.

1.1 Persistent Storage
---------------------------
The recommended way to deploy `YataiService` is with a PostgreSQL DB instance within the cluster, backed with a Persistent Volume. The Helm chart makes it really easy to go this route. All the configuration for this can be found under the Postgres block,

.. code-block:: yaml

    # in BentoML/helm/YataiService/values.yaml

    # Local Postgres Service
    postgres:
        enabled: false # just change this to true
        port: "5432"
        image:
            repo: postgres
            version: latest
            pullPolicy: IfNotPresent
        data:
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: password123
        storageCapacity: 5Gi
        pvReclaimPolicy: Retain

1.2 Ingress
-----------
An Ingress helps to manage incoming external traffic into the cluster. The configuration for this can be found under the ingress block,

.. code-block:: yaml

    # in BentoML/helm/YataiService/values.yaml

    ingress:
        enabled: false
        hostname: {}
        tls:
            enabled: false
            secretName: {}

1.3 TLS
-------
To secure your ingress with TLS, you can enable it by setting `ingress.tls.enabled` to `true`.

.. note::
   When enabling TLS, the Ingress must also be enabled. Please keep in mind that you will need to have a TLS private key and certificate for your hostname if you choose to deploy it.

For local development, create a self-signed key as follows.

.. code-block:: bash

    $ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ${KEY_FILE} -out ${CERT_FILE} -subj "/CN=${HOST}/O=${HOST}"
    Generating a 2048 bit RSA private key
    ...
    writing new private key to 'tls.key'
    -----

    $ kubectl create secret tls ${CERT_NAME} --key ${KEY_FILE} --cert ${CERT_FILE}



Read more here: https://kubernetes.github.io/ingress-nginx/user-guide/tls/

2. Deploying
============

2.1 Minikube (Local)
---------------------------
Minikube is a tool that lets developers run a small Kubernetes cluster on their local machine. Get Minikube here: https://kubernetes.io/docs/tasks/tools/install-minikube/

Then, start a local K8s cluster running by doing `minikube start`.

.. note::

    Before installing the chart, make sure to fetch the `ingress-nginx` dependency by doing `make helm-deps`

Dry Run
~~~~~~~
Developers can do a dry run of the helm chart installation to see if the configuration is valid.

.. code-block:: bash

    $ helm install --dry-run yatai-service helm/YataiService

    NAME: yatai-service
    LAST DEPLOYED: Tue Aug 11 22:12:18 2020
    NAMESPACE: default
    STATUS: pending-install
    REVISION: 1
    TEST SUITE: None
    HOOKS:
    MANIFEST:
    ---
    ...

Then, to install it into the cluster,

.. code-block:: bash

    $ helm install yatai-service helm/YataiService

    NAME: yatai-service
    LAST DEPLOYED: Tue Aug 11 22:18:02 2020
    NAMESPACE: default
    STATUS: deployed
    REVISION: 1
    TEST SUITE: None

    $ kubectl get pods
    NAME                             READY   STATUS    RESTARTS   AGE
    yatai-service-85898d6c9c-ndlfg   1/1     Running   0          91s

After this step, the service should be healthy. Visit the web UI by telling `minikube` to tunnel all of the ports that were defined earlier to your local machine. This should open 2 browser tabs.

.. code-block:: bash

    $ minikube service yatai-service
    |-----------|---------------|-------------|-------------------------|
    | NAMESPACE |     NAME      | TARGET PORT |           URL           |
    |-----------|---------------|-------------|-------------------------|
    | default   | yatai-service | web/3000    | http://172.17.0.3:31891 |
    |           |               | grpc/50051  | http://172.17.0.3:31368 |
    |-----------|---------------|-------------|-------------------------|
    🏃  Starting tunnel for service yatai-service.
    |-----------|---------------|-------------|------------------------|
    | NAMESPACE |     NAME      | TARGET PORT |          URL           |
    |-----------|---------------|-------------|------------------------|
    | default   | yatai-service |             | http://127.0.0.1:56121 |
    |           |               |             | http://127.0.0.1:56122 |
    |-----------|---------------|-------------|------------------------|
    🎉  Opening service default/yatai-service in default browser...
    🎉  Opening service default/yatai-service in default browser...
    ❗  Because you are using a Docker driver on darwin, the terminal needs to be open to run it.

Woo! You now have a `YataiService` instance running on a local K8s cluster :) To cleanup, run `helm uninstall yatai-service` then `minikube stop`.

Keep reading for more info on configuring `YataiService` using Helm.

Custom Values
~~~~~~~~~~~~~

To deploy a `YataiService` instance with a PostgreSQL DB instance within the cluster, developers can use the custom values found in `helm/YataiService/values.yaml`. If this doesn't match the your needs, feel free to create your own custom values files to configure `YataiService` in a way that works for you/your company.

To tell Helm to use these custom values,

.. code-block:: bash

    $ cd helm && helm install -f YataiService/values/postgres.yaml --dry-run --debug yatai-service YataiService

    NAME: yatai-service
    LAST DEPLOYED: Tue Aug 11 22:39:12 2020
    NAMESPACE: default
    STATUS: pending-install
    REVISION: 1
    TEST SUITE: None
    USER-SUPPLIED VALUES:
    db_url: postgresql://postgres:password123@yatai-postgres:5432/postgres
    postgres:
    enabled: true
    ...

You can see a full example K8s manifest here: https://ctrl-v.app/4X2hf7h

If the configuration looks correct, apply it by removing the `--dry-run` and `--debug` flags. Alternatively, run `make helm-install`. Let's double check everything started up correctly.

.. code-block:: bash

    kubectl get all
    NAME                                 READY   STATUS    RESTARTS   AGE
    pod/yatai-postgres-5649dd765c-9c4sp  1/1     Running   0          3s
    pod/yatai-service-556487fb55-wbjc4   1/1     Running   0          3s

    NAME                    TYPE           CLUSTER-IP       EXTERNAL-IP   PORT(S)                          AGE
    service/kubernetes      ClusterIP      10.96.0.1        <none>        443/TCP                          22h
    service/yatai-postgres  NodePort       10.99.249.0      <none>        5432:30007/TCP                   3s
    service/yatai-service   LoadBalancer   10.107.204.236   <pending>     3000:32422/TCP,50051:30014/TCP   3s

    NAME                            READY   UP-TO-DATE   AVAILABLE   AGE
    deployment.apps/yatai-postgres  1/1     1            1           3s
    deployment.apps/yatai-service   1/1     1            1           3s

    NAME                                       DESIRED   CURRENT   READY   AGE
    replicaset.apps/yatai-postgres-5649dd765c  1         1         1       3s
    replicaset.apps/yatai-service-556487fb55   1         1         1       3s

Everything looks good!

2.2 Cloud Providers
-------------------
This part of the BentoML documentation is a work in progress. If you have any questions
related to this, please join
`the BentoML Slack community <https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg>`_
and ask in the bentoml-users channel.
