==========================
Monitoring with Prometheus
==========================

Monitoring stacks usually consist of a metrics collector, a time-series database to store metrics
and a visualization layer. A popular stack is `Prometheus <https://prometheus.io/>`_ with `Grafana <https://grafana.com/>`_
used as the visualization layer to create rich dashboards. An architecture of Prometheus and its ecosystem is shown below:

.. image:: ../_static/img/prom-architecture.png


BentoML API server comes with Prometheus support out of the box. When launching an API model server with BentoML,
whether it is running dev server locally or deployed with docker in the cloud, a ``/metrics`` endpoint will always
be available for exposing prometheus metrics. This guide will introduce how you use Prometheus and Grafana to monitor
your BentoService.

Preface
-------

.. seealso::
    `Prometheus Docs <https://prometheus.io/docs/introduction/overview/>`_ and `Grafana Docs <https://grafana.com/docs/grafana/latest/>`_ for more in depth topics.

.. note::
    This guide requires users to have a basic understanding of Prometheus' concept as well as its metrics type. Please refers
    to `Concepts <https://prometheus.io/docs/concepts/data_model/>`_ for more information.

.. note::
    Prometheus comes with a query language called PromQL. Refers to `query basics <https://prometheus.io/docs/prometheus/latest/querying/basics/>`_.

.. note::
    Please refers to Prometheus' best practices for `consoles and dashboards <https://prometheus.io/docs/practices/consoles/>`_
    as well as `histogram and summaries <https://prometheus.io/docs/practices/histograms/>`_.

Setting up Prometheus
---------------------

It is recommended to run Prometheus with Docker . Please make sure that you have
`Docker <https://docs.docker.com/engine/install/>`_ installed on your system.

Users can take advantage of having a ``prometheus.yml`` for configuration.
Example of for monitoring BentoService is shown below:

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
        - localhost:9090 # metrics from prom/prometheus
        - localhost:5000 # BentoService metrics
        - localhost:3000 # YataiService metrics

.. note::
    Make sure you have targets port for your BentoService setup correctly according to your use case. For the
    demo we will use BentoService default port **5000**.


We can then run Prometheus with the following:

.. code-block:: bash

    # Bind-mount your prometheus.yml from the host by running:
    » docker run --network=host -v path/to/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

.. note::
    When deploying users can setup ``docker-compose`` and setup a shared network in order for ``prometheus`` to scrape
    metrics from your BentoService. Please refers to :ref:`docker-compose`.

Users can check `<http://localhost:9090/status>`_ to make sure prometheus is currently running.
In order to check if prometheus is scraping our BentoService, `<http://localhost:9090/targets>`_ should show:

.. image:: ../_static/img/prom-targets-running.png

Setting up Grafana
------------------

It is recommend to also use Grafana with Docker.

.. code-block:: bash

    » docker run --network=host grafana/grafana

To log in to Grafana for the first time:

    #. Open your web browser and go to http://localhost:3000/. The default HTTP port that Grafana listens to is 3000 unless you have configured a different port.

    #. On the login page, enter ``admin`` for username and password.

    #. Click Log in. If login is successful, then you will see a prompt to change the password.

    #. Click OK on the prompt, then change your password.

.. seealso::
    `Add Prometheus Datasource on Grafana <https://grafana.com/docs/grafana/latest/datasources/prometheus/>`_.

.. seealso::::
    `Best practice while creating dashboards on Grafana <https://grafana.com/docs/grafana/latest/best-practices/best-practices-for-creating-dashboards/>`_

Users can also import `BentoService Dashboard <https://github.com/bentoml/BentoML/tree/master/docs/source/configs/grafana/provisioning/dashboards/bentoml_service-1623377681395.json>`_ and
explore given BentoService metrics by `importing dashboard on Grafana <https://grafana.com/docs/grafana/latest/dashboards/export-import/#import-dashboard>`_.

.. image:: ../_static/img/bentoml-grafana-dashboard.png



======================

.. warning::
    Make sure to setup correctly all dependencies before proceeding: `Docker Swarm <https://docs.docker.com/engine/swarm/swarm-tutorial/#set-up>`_, `kubectl <https://kubernetes.io/docs/tasks/tools/>`_.

.. _docker-compose:
.. include:: configs/README.rst
   :start-after: introduction-marker
   :end-before: not-exposed-marker

content of ``docker-compose.yml``:

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
        image: aarnphm/bentoml-sentiment-analysis:latest
        ports:
          - "5000:5000"
        networks:
          - shared-network
        deploy:
          mode: global
        restart: on-failure

.. seealso::
    `Alertmanager <https://prometheus.io/docs/alerting/latest/alertmanager/>`_ and `cAdvisor <https://github.com/google/cadvisor>`_ to setup alerts as well as monitor container resources.

.. seealso::
    `prom/node-exporter <https://github.com/prometheus/node_exporter>`_ for expose machine metrics.


(Optional) Deploying Prometheus on Kubernetes
---------------------------------------------

.. seealso::
    `configs/deployment <https://github.com/bentoml/BentoML/tree/master/docs/source/guides/configs/deployment>`_ for k8s deployment config.

(Optional) Exposing GPU Metrics on Kubernetes
---------------------------------------------

For monitoring GPU metrics on Kubernetes, it is recommended to use ``dcgm-exporter``.