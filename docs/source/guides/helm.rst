Using Helm to install YataiService
=============================================

*Helm* is a tool that streamlines installing and managing K8s applications. It lets us easily configure a set of K8s resources that make up our application -- YataiService in this case. Helm enables us to

- easily deploy recommended configuration and setup for YataiService
- easily change settings
- drop and replace parts with your own (e.g. replace local Postgres with a remote RDS)

1. Overview of resources
==============================================================
YataiService
PostgresService
Ingress

2. Configuration
==============================================================
Helm charts rely on a values file to configure how to template and create your resources. You can find the default values in `BentoML/helm/YataiService/values.yaml`. If you wish to modify more values, you can create a file under `BentoML/helm/YataiService/values/<name>.yaml`. You can find more details about deployment in the next section.

2.1 Persistent Storage
---------------------------

2.2 Ingress
---------------------------
An Ingress helps to manage incoming external traffic into your cluster. The configuration for this can be found under the ingress block,

.. code-block:: yaml

    ingress:
        enabled: false
        hostname: {}
        tls:
            enabled: false
            secretName: {}

2.3 TLS
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

3. Deploying
==============================================================
3.1 Minikube (Local)
---------------------------
<install minikube>
<install helm>

3.2 Google Kubernetes Engine
----------------------------
This part of the BentoML documentation is a work in progress. If you have any questions
related to this, please join
`the BentoML Slack community <https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg>`_
and ask in the bentoml-users channel.
