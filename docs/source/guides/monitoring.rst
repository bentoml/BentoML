==========================
Monitoring with Prometheus
==========================

Monitoring stacks usually consist of a metrics collector, a time-series database to store metrics
and a visualization layer. A popular open-source stack is Prometheus used with Grafana as visualization tool to create rich dashboards.

BentoML API server comes with Prometheus support out of the box. When launching an API model server with BentoML,
whether it is running dev server locally or deployed with docker in the cloud, a ``/metrics`` endpoint will always
be available, which includes the essential metrics for model serving and the ability to create
and customize new metrics base on needs. This guide will introduce how to use Prometheus and Grafana to monitor
your BentoService.

Preface
-------

.. seealso::
     Prometheus_ and Grafana_ docs for more in depth topics.

.. note::
    This guide requires users to have a basic understanding of Prometheus' concept as well as its metrics type.
    Please refers to `Concepts <https://prometheus.io/docs/concepts/data_model/>`_ for more information.

.. note::
    Refers to `PromQL basics <https://prometheus.io/docs/prometheus/latest/querying/basics/>`_ for Prometheus query language.

.. note::
    Please refers to Prometheus' best practices for `consoles and dashboards <https://prometheus.io/docs/practices/consoles/>`_
    as well as `histogram and summaries <https://prometheus.io/docs/practices/histograms/>`_.

.. note::
    Users can also create custom metrics for *BentoService* making use of |prom_client|_, which can be later scraped by Prometheus.

    .. code-block:: python

        from bentoml.configuratoin.containers import BentoMLContainer

	metrics_client = BentoMLContainer.metircs_client.get()

        REQUEST_TIME = metrics_clint.Summary('request_processing_time', 'Time spend processing request')

        @artifacts([KerasModelArtifact('model'), PickleArtifact('tokenizer')])
        class TensorflowService(BentoService):

            @REQUEST_TIME.time()
            @api(input=JsonInput())
            def predict(self, parsed_json):
                raw = self.preprocessing(parsed_json['text'])
                input_data = [raw[: n + 1] for n in range(len(raw))]
                input_data = pad_sequences(input_data, maxlen=100, padding="post")
                return self.artifacts.model.predict(input_data)


Local Deployment
----------------

This section will walk you through how to set up the stack locally, with the optional guide on using ``docker-compose`` for easy deployment of the stack.


Setting up Prometheus
^^^^^^^^^^^^^^^^^^^^^

It is recommended to run Prometheus with Docker. Please make sure that you have
`Docker <https://docs.docker.com/engine/install/>`_ installed on your system.

Users can take advantage of having a ``prometheus.yml`` for configuration.
An example to monitor multiple BentoServices is shown below:

.. code-block:: yaml

    # prometheus.yml

    global:
      scrape_interval:     15s
      evaluation_interval: 30s
      # scrape_timeout is set to the global default (10s).

    scrape_configs:
    - job_name: prometheus

      honor_labels: true
      static_configs:
      - targets:
        - localhost:5000  # metrics from SentimentClassifier service
        - localhost:6000  # metrics from IrisClassifier service

.. note::
    In order to monitor multiple BentoServices, make sure to set up different ports for each BentoService and add
    correct targets under ``static_configs`` as shown above.


We can then run Prometheus with the following:

.. code-block:: bash

    # Bind-mount your prometheus.yml from the host by running:
    » docker run --network=host -v path/to/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

.. note::
    When deploying, users can set up ``docker-compose`` and a shared network in order for ``prometheus`` to scrape
    metrics from your BentoService. Please refers to :ref:`docker-compose`.

Users can check |localhost-9090|_ to make sure prometheus is running.
In order to check if prometheus is scraping our BentoService, |9090-target|_ should show:

.. _localhost-9090: http://localhost:9090/status
.. |localhost-9090| replace:: ``:9090/status``

.. _9090-target: http://localhost:9090/targets
.. |9090-target| replace:: ``:9090/targets``

.. image:: ../_static/img/prom-targets-running.png

Setting up Grafana
^^^^^^^^^^^^^^^^^^

It is also recommended to use Grafana with Docker.

.. code-block:: bash

    » docker run --network=host grafana/grafana

To log in to Grafana for the first time:

    #. Open your web browser and go to |localhost-3000|_. The default HTTP port that Grafana listens to is ``:3000`` unless you have configured a different port.

    #. On the login page, enter ``admin`` for username and password.

    #. Click Log in. If login is successful, you will see a prompt to change the password.

    #. Click OK on the prompt, then change your password.

.. _localhost-3000: http://localhost:3000
.. |localhost-3000| replace:: **localhost:3000**

.. seealso::
    `Add Prometheus Datasource on Grafana <https://grafana.com/docs/grafana/latest/datasources/prometheus/>`_.

.. seealso::::
    `Best practice while creating dashboards on Grafana <https://grafana.com/docs/grafana/latest/best-practices/best-practices-for-creating-dashboards/>`_

Users can also import `BentoService Dashboard <https://github.com/bentoml/BentoML/blob/master/docs/source/guides/configs/grafana/provisioning/dashboards/bentoml-dashboard-docker-swarm-1623677587070.json>`_ and
explore given BentoService metrics by `importing dashboard <https://grafana.com/docs/grafana/latest/dashboards/export-import/#import-dashboard>`_.

.. image:: ../_static/img/bentoml-grafana-dashboard.png


======================

.. warning::
    Make sure to set up `Docker Swarm <https://docs.docker.com/engine/swarm/swarm-tutorial/#set-up>`_ before proceeding.

.. _docker-compose:
.. include:: configs/README.rst
   :start-after: introduction-marker
   :end-before: not-exposed-marker

content of ``docker-compose.yml``, a sample dashboard can be seen `here <https://snapshot.raintank.io/dashboard/snapshot/yZQSlxxQ7VPjBmTbnGOUxVyIkpLJqnU5>`_:

.. code-block:: yaml

    version: '3.7'

    volumes:
      prometheus_data:
      grafana_data:

    networks:
      shared-network:

    services:

      prometheus:
        image: prom/prometheus
        volumes:
          - ./prometheus/:/etc/prometheus/
          - prometheus_data:/prometheus
        command:
          - '--config.file=/etc/prometheus/prometheus.yml'
          - '--storage.tsdb.path=/prometheus'
          - '--web.console.libraries=/usr/share/prometheus/console_libraries'
          - '--web.console.templates=/usr/share/prometheus/consoles'
        ports:
          - 9090:9090
        networks:
          - shared-network
        deploy:
          placement:
            constraints:
              - node.role==manager
        restart: on-failure

      grafana:
        image: grafana/grafana
        depends_on:
          - prometheus
        ports:
          - 3000:3000
        volumes:
          - grafana_data:/var/lib/grafana
          - ./grafana/provisioning/:/etc/grafana/provisioning/
        env_file:
          - ./grafana/config.monitoring
        networks:
          - shared-network
        user: "472"
        deploy:
          mode: global
        restart: on-failure

      bentoml:
        image: bentoml/fashion-mnist-classifier:latest
        ports:
          - "5000:5000"
        networks:
          - shared-network
        deploy:
          mode: global
        restart: on-failure

.. seealso::
    `Alertmanager <https://prometheus.io/docs/alerting/latest/alertmanager/>`_ and `cAdvisor <https://github.com/google/cadvisor>`_ to set up alerts as well as monitor container resources.

.. seealso::
    `prom/node-exporter <https://github.com/prometheus/node_exporter>`_ for expose machine metrics.


==============================

Deploy on Kubernetes
-----------------------

.. note::
    `minikube <https://minikube.sigs.k8s.io/docs/start/>`_ and `kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/>`_ are required for this part of the tutorial.
    Users may also choose to install `virtualbox <https://www.virtualbox.org/>`_ in order to run minikube.

.. seealso::
    :doc:`../deployment/kubernetes` on how to deploy BentoService to Kubernetes.


Deploy Prometheus on K8s
^^^^^^^^^^^^^^^^^^^^^^^^

Setting Prometheus stack on Kubernetes could be an arduous task. However, we can take advantage of ``Helm`` package manager
and make use of `prometheus-operator <https://github.com/prometheus-operator/prometheus-operator>`_ through kube-prometheus_:

* The Operator uses standard configurations and dashboards for Prometheus and Grafana.
* The Helm ``prometheus-operator`` chart allows you to get a full cluster monitoring solution up and running by installing aforementioned components.

.. seealso::
    kube-prometheus_

.. warning::
    Your local minikube cluster will be delete in order to set up ``kube-prometheus-stack`` correctly.

Set up virtualbox to be default driver for ``minikube``:

.. code-block:: bash

    » minikube config set driver virtualbox

Spin up our local K8s cluster:

.. code-block:: bash

    # prometheus-operator/kube-prometheus
    » minikube delete && minikube start \
        --kubernetes-version=v1.20.0 \
        --memory=6g --bootstrapper=kubeadm \
        --extra-config=kubelet.authentication-token-webhook=true \
        --extra-config=kubelet.authorization-mode=Webhook \
        --extra-config=scheduler.address=0.0.0.0 \
        --extra-config=controller-manager.address=0.0.0.0

.. note::
    We allocate 6Gb of memory via ``--memory`` for this K8s cluster.
    Change the value to fit your use-case.

Then get ``helm`` repo:

.. code-block:: bash

    » helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

.. code-block:: bash

    » helm repo update

Search for available prometheus chart:

.. code-block:: bash

    » helm search repo kube-prometheus

Once located the version, inspect the chart to modify the settings:

.. code-block:: bash

    » helm inspect values prometheus-community/kube-prometheus-stack \
        > ./configs/deployment/kube-prometheus-stack.values


Next, we need to change Prometheus server service type in order to access it from the browser,
by changing our service type from ``ClusterIP`` to ``NodePort``. This enables Prometheus server to be accessible at your machine ``:30090``

.. code-block:: yaml

      ## Configuration for Prometheus service
      ##
      service:
        annotations: {}
        labels: {}
        clusterIP: ""

        ## Port for Prometheus Service to listen on
        ##
        port: 9090

        ## To be used with a proxy extraContainer port
        targetPort: 9090

        ## List of IP addresses at which the Prometheus server service is available
        ## Ref: https://kubernetes.io/docs/user-guide/services/#external-ips
        ##
        externalIPs: []

        ## Port to expose on each node
        ## Only used if service.type is 'NodePort'
        ##
        nodePort: 30090

        ## LoadBalancer IP
        ## Only use if service.type is "LoadBalancer"
        loadBalancerIP: ""
        loadBalancerSourceRanges: []
        ## Service type
        ##
        type: NodePort # changed this line from ClusterIP to NodePort

By default, Prometheus discovers |PodMonitors|_ and |ServiceMonitors|_ within its namespace, with same release tags labeled as ``prometheus-operator`` release.
Since we want to Prometheus to discover our BentoService (refers to :ref:`custom-service-monitor`), we need to create a custom PodMonitors/ServiceMonitors to scrape metrics from our services. Thus, one
way to do this is to allow Prometheus to discover all PodMonitors/ServiceMonitors within its name, without applying label filtering. Set the following options:

.. code-block:: yaml

    - prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues: false
    - prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues: false

.. _PodMonitors: https://prometheus-operator.dev/docs/operator/api/#podmonitor
.. |PodMonitors| replace:: PodMonitors
.. _ServiceMonitors: https://prometheus-operator.dev/docs/operator/api/#servicemonitor
.. |ServiceMonitors| replace:: ServiceMonitors


Finally deploy Prometheus and Grafana pods using ``kube-prometheus-stack`` via Helm:

.. code-block:: bash

    » helm install prometheus-community/kube-prometheus-stack \
        --create-namespace --namespace bentoml \
        --generate-name --values ./configs/deployment/kube-prometheus-stack.values

.. code-block:: bash

    NAME: kube-prometheus-stack-1623502925
    LAST DEPLOYED: Sat Jun 12 20:02:09 2021
    NAMESPACE: bentoml
    STATUS: deployed
    REVISION: 1
    NOTES:
    kube-prometheus-stack has been installed. Check its status by running:
      kubectl --namespace bentoml get pods -l "release=kube-prometheus-stack-1623502925"

    Visit https://github.com/prometheus-operator/kube-prometheus for instructions on how to create & configure Alertmanager and Prometheus instances using the Operator.

.. note::
    You can also provides the values in chart directly with helm ``--set``, e.g:

    * ``--set prometheus.service.type=NodePort``

    * ``--set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false``

    * ``--set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false``


Check for Prometheus and Grafana pods:

.. code-block:: bash

    » kubectl get pods -A

.. code-block:: bash

    NAMESPACE     NAME                                                              READY   STATUS    RESTARTS   AGE
    bentoml       kube-prometheus-stack-1623-operator-5555798f4f-nghl8              1/1     Running   0          4m22s
    bentoml       kube-prometheus-stack-1623502925-grafana-57cdffccdc-n7lpk         2/2     Running   0          4m22s
    bentoml       prometheus-kube-prometheus-stack-1623-prometheus-0                2/2     Running   1          4m5s

Check for service startup as part of the operator:

.. code-block:: bash

    » kubectl get svc -A

.. code-block:: bash

    NAMESPACE     NAME                                                        TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)                        AGE
    bentoml       alertmanager-operated                                       ClusterIP   None             <none>        9093/TCP,9094/TCP,9094/UDP     5m8s
    bentoml       kube-prometheus-stack-1623-operator                         ClusterIP   10.106.5.23      <none>        443/TCP                        5m25s
    bentoml       kube-prometheus-stack-1623-prometheus                       NodePort    10.96.241.205    <none>        9090:30090/TCP                 5m26s
    bentoml       kube-prometheus-stack-1623502925-grafana                    ClusterIP   10.111.205.42    <none>        80/TCP                         5m25s
    bentoml       prometheus-operated                                         ClusterIP   None             <none>        9090/TCP                       5m8s

As we can observe that the Prometheus server is available at ``:30090``. Thus, open browser at ``http://<machine-ip-addr>:30090``.
By default the Operator enables users to monitor our Kubernetes cluster.

Using Grafana
^^^^^^^^^^^^^

Users can also launch the Grafana tools for visualization.

There are two ways to deal with exposing Grafana ports, either is recommended based on preference:

* :ref:`patching`.

* :ref:`port-forwarding` `(Official guides) <https://github.com/prometheus-operator/kube-prometheus#access-the-dashboards>`_.

.. _patching:

Patching Grafana Service
""""""""""""""""""""""""

By default, Every services in the Operator uses ``ClusterIP`` to expose the ports where the service is accessible, including
Grafana. This can be changed to a ``NodePort`` instead, so the page is accessible from the browser, similar to what we did earlier
with Prometheus dashboard.

We can take advantage of |kubectl_patch|_ to update the service API to expose a ``NodePort`` instead.

.. _kubectl_patch: https://kubernetes.io/docs/tasks/manage-kubernetes-objects/update-api-object-kubectl-patch/
.. |kubectl_patch| replace:: **kubectl patch**

Modify the spec to change service type:

.. code-block:: bash

    » cat << EOF | tee ./configs/deployment/grafana-patch.yaml
    spec:
      type: NodePort
      nodePort: 36745
    EOF

Use ``kubectl patch``:

.. code-block:: bash

    # This is how we get grafana pod name

    » _GRAFANA_SVC=$(kubectl get svc -n bentoml | grep grafana | cut -d " " -f1)

.. code-block:: bash

    » kubectl patch svc "${_GRAFANA_SVC}" -n bentoml --patch "$(cat configs/deployment/grafana-patch.yaml)"

    service/kube-prometheus-stack-1623502925-grafana patched

Verify that the service is now exposed at an external accessible port:

.. code-block:: bash

    » kubectl get svc -A

.. code-block:: bash

    NAMESPACE    NAME                                          TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)          AGE
    bentoml      kube-prometheus-stack-1623-prometheus         NodePort    10.96.241.205    <none>        9090:30090/TCP   35m
    bentoml      kube-prometheus-stack-1623502925-grafana      NodePort    10.111.205.42    <none>        80:32447/TCP     35m

Open your browser at ``http:<machine-ip-addr>:32447``, credentials:

* login: ``admin``

* password: ``prom-operator``.


.. _port-forwarding:

Port Forwarding
"""""""""""""""

Another method is to access Grafana with port-forwarding.

Notice that Grafana is accessible at port ``:80``. We will choose an arbitrary port ``:36745`` on our local machine to port ``:80`` on the service (-> ``:3000`` where
Grafana is listening at)

.. code-block:: bash

    » kubectl port-forward svc/kube-prometheus-stack-1623502925-grafana -n bentoml 36745:80

    Forwarding from 127.0.0.1:36745 -> 3000
    Forwarding from [::1]:36745 -> 3000
    Handling connection for 36745

.. note::
    If your cluster is set up on a cloud instance, e.g. AWS EC2, you might have to set up SSH tunnel between your local
    workstation and the instance using port forwarding to view Grafana tool in your own browser.

Point to ``http://localhost:36745/`` to see Grafana login page using the same credentials.

.. _custom-service-monitor:

Setting up your BentoService
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example BentoService with custom ServiceMonitor on Kubernetes:

.. code-block:: yaml

    ---
    ### BentoService
    apiVersion: v1
    kind: Service
    metadata:
      labels:
        app: bentoml-service
      name: bentoml-service
      namespace: bentoml
    spec:
      externalTrafficPolicy: Cluster
      ports:
        - name: predict
          nodePort: 32610
          port: 5000
          protocol: TCP
          targetPort: 5000
      selector:
        app: bentoml-service
      sessionAffinity: None
      type: NodePort

    ---
    ### BentoService ServiceMonitor
    apiVersion: monitoring.coreos.com/v1
    kind: ServiceMonitor
    metadata:
      name: bentoml-service
      namespace: bentoml
    spec:
      selector:
        matchLabels:
          app: bentoml-service
      endpoints:
      - port: predict

.. note::
    Make sure that you also include a custom ServiceMonitor definition for your BentoService. For information on
    how to use ServiceMonitor CRD, please see the `documentation <https://github.com/prometheus-operator/prometheus-operator/blob/master/Documentation/user-guides/getting-started.md#include-servicemonitors>`_.

Apply the changes to enable monitoring:

.. code-block:: bash

    kubectl apply -f configs/deployment/bentoml-deployment.yml --namespace=bentoml

.. note::
    After logging into Grafana, imports the provided kubernetes dashboards under ``configs/grafana/provisioning/dashboards``.

**The final results:** Deployed BentoML-Prometheus-Grafana Stack on Kubernetes

.. code-block:: bash

    » minikube service list
    |-------------|-----------------------------------------------------------|--------------|-----------------------------|
    |  NAMESPACE  |                           NAME                            | TARGET PORT  |             URL             |
    |-------------|-----------------------------------------------------------|--------------|-----------------------------|
    | bentoml     | bentoml-service                                           | predict/5000 | http://192.168.99.103:32610 |
    | bentoml     | kube-prometheus-stack-1623-prometheus                     | web/9090     | http://192.168.99.102:30090 |
    | bentoml     | kube-prometheus-stack-1623502925-grafana                  | service/80   | http://192.168.99.102:32447 |
    |-------------|-----------------------------------------------------------|--------------|-----------------------------|

.. image:: ../_static/img/k8s-bentoml.png

.. image:: ../_static/img/k8s-grafana.png

.. image:: ../_static/img/k8s-prometheus.png

.. note::
    You might have to wait a few minutes for everything to spin up. In the meantime, an example dashboard on `Kubernetes <https://snapshot.raintank.io/dashboard/snapshot/qZRYdOb5UqmrZ4QU5RspbJeZOqkOJYLH>`_.
    You can check your namespace pods health with ``minikube dashboard``:

    .. image:: ../_static/img/k8s-minikube.png


.. note::
    Mounting `PersistentVolume` for Prometheus and Grafana on K8s is currently working in progress.

(Optional) Exposing GPU Metrics on Kubernetes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: This part is currently working in progress. If you have any questions related to this, please join `the BentoML Slack community <https://join.slack.com/t/bentoml/shared_invite/enQtNjcyMTY3MjE4NTgzLTU3ZDc1MWM5MzQxMWQxMzJiNTc1MTJmMzYzMTYwMjQ0OGEwNDFmZDkzYWQxNzgxYWNhNjAxZjk4MzI4OGY1Yjg>`_
    and ask in the bentoml-users channel.

.. _kube-prometheus: https://github.com/prometheus-operator/kube-prometheus#readme
.. _Prometheus: https://prometheus.io/docs/introduction/overview/
.. _Grafana: https://grafana.com/docs/grafana/latest/

.. spelling::

    Yatai
    tsdb
    Alertmanager
    virtualbox
    Gb
    repo
    datasources
