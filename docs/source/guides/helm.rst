Using Helm to install YataiService
=============================================

*Helm* is a tool that streamlines installing and managing K8s applications. It lets us easily configure a set of K8s resources that make up our application -- YataiService in this case. Helm enables us to

- easily deploy recommended configuration and setup for YataiService
- easily change settings
- drop and replace parts with your own (e.g. replace local Postgres with a remote RDS)

1. Configuration
==============================================================
Helm charts rely on a values file to configure how to template and create your resources. You can find the default values in `BentoML/helm/YataiService/values.yaml`. These basic values describe a basic local `YataiService` instance with ephemereal storage.

1.1 Persistent Storage
---------------------------
The recommended way to deploy `YataiService` is with a PostgreSQL DB instance within the cluster, backed with a Persistent Volume. The Helm chart makes it really easy to go this route. You can find all the configuration for this under the Postgres block,

.. code-block:: yaml

    # in BentoML/helm/YataiService/values.yaml

    # Local Postgres Service
    postgres:
    enabled: false # just change this to true
    port: "5432"
    image:
        repo: postgres
        version: 10.4
        pullPolicy: IfNotPresent
    data:
        POSTGRES_DB: bentomldb
        POSTGRES_USER: postgresadmin
        POSTGRES_PASSWORD: admin123
    storageCapacity: 5Gi

1.2 Ingress
---------------------------
An Ingress helps to manage incoming external traffic into your cluster. The configuration for this can be found under the ingress block,

.. code-block:: yaml

    # in BentoML/helm/YataiService/values.yaml

    ingress:
        enabled: false
        hostname: {}
        tls:
            enabled: false
            secretName: {}

1.3 TLS
---------------------------
If you would like to secure your ingress with TLS, you can enable it by setting `ingress.tls.enabled` to `true`.

.. note::
   If you plan on enabling TLS, the Ingress must be enabled as well. Please keep in mind that you will need to have a TLS private key and certificate for your hostname if you choose to deploy it.

For local development, you can create a self-signed key as follows.

.. code-block:: bash

    $ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ${KEY_FILE} -out ${CERT_FILE} -subj "/CN=${HOST}/O=${HOST}"
    Generating a 2048 bit RSA private key
    ...
    writing new private key to 'tls.key'
    -----

    $ kubectl create secret tls ${CERT_NAME} --key ${KEY_FILE} --cert ${CERT_FILE}



Read more here: https://kubernetes.github.io/ingress-nginx/user-guide/tls/

2. Deploying
==============================================================
2.1 Minikube (Local)
---------------------------
Minikube is a tool that lets you run a small Kubernetes cluster on your local machine. Recommended for testing. You can get Minikube here: https://kubernetes.io/docs/tasks/tools/install-minikube/

Helm is a CLI tool that helps us define, configure, and install K8s applications. Install it here: https://helm.sh/docs/intro/install/

Then, make sure you have a local K8s cluster running by doing `minikube start`.

=======
Dry Run
=======
Let's do a dry run of the helm chart installation to see if our configuration is valid.

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

Looks like all of the resource we need to deploy are all there! Let's install it into our cluster.

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

Awesome! The service is healthy. Let's check out the web UI by telling `minikube` to tunnel all of the ports that we defined earlier to our local machine. This should open 2 browser tabs.

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

Woo! You now have `YataiService` running on a local K8s cluster :) To cleanup, run `helm uninstall yatai-service` then `minikube stop`.

Keep reading for more info on configuring `YataiService` using Helm.

=======
Custom Values
=======

Now, say you wanted to deploy `YataiService` with a a PostgreSQL DB instance within the cluster. We created a custom values file just for this reason. You can find it in `helm/YataiService/values/postgres.yaml`. Feel free to create your own custom values files to configure `YataiService` in a way that works for you/your company.

To tell Helm to use these custom values, we can do this

.. code-block:: bash

    $ cd helm && helm install -f YataiService/values/postgres.yaml --dry-run --debug yatai-service YataiService

    NAME: yatai-service
    LAST DEPLOYED: Tue Aug 11 22:39:12 2020
    NAMESPACE: default
    STATUS: pending-install
    REVISION: 1
    TEST SUITE: None
    USER-SUPPLIED VALUES:
    db_url: postgresql://postgresadmin:admin123@postgres:5432/bentomldb
    postgres:
    enabled: true
    ...

Or, if you prefer a shortcut, `make helm-dry`. You can see a full example K8s manifest here: https://ctrl-v.app/25OF7eK.

Now that we've done a dry-run and we're happy with the resources Helm plans on creating, let's apply it by removing the `--dry-run` and `--debug` flags. Alternatively, you can run `make helm-install`. Let's double check everything started up correctly.

.. code-block:: bash

    kubectl get all
    NAME                                 READY   STATUS    RESTARTS   AGE
    pod/postgres-5649dd765c-9c4sp        1/1     Running   0          3s
    pod/yatai-service-556487fb55-wbjc4   1/1     Running   0          3s

    NAME                    TYPE           CLUSTER-IP       EXTERNAL-IP   PORT(S)                          AGE
    service/kubernetes      ClusterIP      10.96.0.1        <none>        443/TCP                          22h
    service/postgres        NodePort       10.99.249.0      <none>        5432:30007/TCP                   3s
    service/yatai-service   LoadBalancer   10.107.204.236   <pending>     3000:32422/TCP,50051:30014/TCP   3s

    NAME                            READY   UP-TO-DATE   AVAILABLE   AGE
    deployment.apps/postgres        1/1     1            1           3s
    deployment.apps/yatai-service   1/1     1            1           3s

    NAME                                       DESIRED   CURRENT   READY   AGE
    replicaset.apps/postgres-5649dd765c        1         1         1       3s
    replicaset.apps/yatai-service-556487fb55   1         1         1       3s

Everything looks good! Enjoy your new `YataiService` cluster :))

2.2 Cloud Providers
----------------------------
This part of the BentoML documentation is a work in progress. If you have any questions
related to this, please join
`the BentoML Slack community <https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg>`_
and ask in the bentoml-users channel.
