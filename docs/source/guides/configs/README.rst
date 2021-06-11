.. introduction-marker

(Optional) Prometheus - Grafana and *docker-compose* stack
----------------------------------------------------------

Users can freely update `prometheus.yml <https://github.com/bentoml/BentoML/tree/master/docs/source/guides/configs/prometheus/prometheus.yml>`_ target section to define what should be  monitored by Prometheus.

`grafana/provisioning <https://github.com/bentoml/BentoML/tree/master/docs/source/guides/configs/grafana/provisioning>`_ provides both `datasources` and `dashboards` for us to specify datasources and bootstrap our dashboards quickly courtesy of the `introduction of Provisioning in v5.0.0 <https://grafana.com/docs/grafana/latest/administration/provisioning/>`_

If you would like to automate the installation of additional dashboards just copy the Dashboard JSON file to `grafana/provisioning/dashboards <https://github.com/bentoml/BentoML/tree/master/docs/source/guides/grafana/provisioning/dashboards>`_ and it will be provisioned next time you stop and start Grafana.

.. seealso::
    `Config Project Structure <https://github.com/bentoml/BentoML/tree/master/docs/source/guides/configs>`_

    .. code-block:: bash

        .
        ├── deployment
        ├── grafana
        │   ├── config.monitoring
        │   └── provisioning
        │       ├── dashboards
        │       └── datasources
        ├── prometheus
        │   ├── alert.rules
        │   └── prometheus.yml
        ├── docker-compose.yml
        └── README.rst

.. not-exposed-marker

Pre-requisite
-------------

    * `Docker Swarm <https://docs.docker.com/engine/swarm/>`_ (bundled with Docker Desktop Mac/Windows)

    * `docker_compose <https://docs.docker.com/compose/install>`_

    .. code-block:: bash

        » sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && sudo chmod +x /usr/local/bin/docker-compose