.. introduction-marker

(Optional) Prometheus - Grafana and *docker-compose* stack
----------------------------------------------------------

Users can freely update `prometheus.yml <https://github.com/bentoml/BentoML/tree/master/docs/source/guides/configs/prometheus/prometheus.yml>`_ target section to define what should be  monitored by Prometheus.

`grafana/provisioning <https://github.com/bentoml/BentoML/tree/master/docs/source/guides/configs/grafana/provisioning>`_ provides both `datasources` and `dashboards` for us to specify datasources and bootstrap our dashboards quickly courtesy of the `introduction of Provisioning in v5.0.0 <https://grafana.com/docs/grafana/latest/administration/provisioning/>`_

If you would like to automate the installation of additional dashboards just copy the Dashboard JSON file to `grafana/provisioning/dashboards <https://github.com/bentoml/BentoML/tree/master/docs/source/guides/grafana/provisioning/dashboards>`_ and it will be provisioned next time you stop and start Grafana.

.. seealso::
    `In-depth guide of using the stack <https://github.com/bentoml/BentoML/tree/master/docs/source/guides/configs>`_

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

Prerequisite
------------

* `Docker Swarm <https://docs.docker.com/engine/swarm/>`_ (bundled with Docker Desktop Mac/Windows)

* `docker-compose <https://docs.docker.com/compose/install>`_

.. code-block:: bash

    » sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

    » sudo chmod +x /usr/local/bin/docker-compose

Installation
------------

Docker Swarm for Linux distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You have to specify ip-ports for ``docker swarm init``. Ex: 2402:800:6837:92f3::f

.. code-block:: bash

    » docker swarm init --advertise-addr <ip-port>

User can choose to either deploy with swarm (recommended) or run ``docker-compose``

Run Locally
^^^^^^^^^^^

.. code-block:: bash

    » docker-compose up --remove-orphans

Deploy with Swarm
^^^^^^^^^^^^^^^^^

.. code-block:: bash

    » docker stack deploy -c docker-compose.yml bento-prom-stack

That's its !!! ``docker stack deploy`` deploys our entire stack automatically to Docker Swarm.

Aftermath
^^^^^^^^^

Grafana should be accessible via ``http://127.0.0.1:3000`` with credentials

::

    user: admin
    passwd: foobar (can be stored under `grafana/config.monitoring <./grafana/config.monitoring>`_)


Check status of our newly created stack:

.. code-block:: bash

    » docker stack ps bento-prom-stack
    ID             NAME                                                 IMAGE                                       NODE        DESIRED STATE   CURRENT STATE           ERROR     PORTS
    vcedt3c3nhpa   bento-prom-stack_bentoml.76q5j547rpxwlkbpqdzuh95ww   aarnphm/bentoml-sentiment-analysis:latest   archlinux   Running         Running 2 minutes ago
    hyhx7fqmw8g8   bento-prom-stack_grafana.76q5j547rpxwlkbpqdzuh95ww   grafana/grafana:latest                      archlinux   Running         Running 2 minutes ago
    yq6xbvwfqkri   bento-prom-stack_prometheus.1                        prom/prometheus:latest                      archlinux   Running         Running 2 minutes ago

View running services:

.. code-block:: bash

    » docker service ls
    ID             NAME                          MODE         REPLICAS   IMAGE                                       PORTS
    qm231pjikabq   bento-prom-stack_bentoml      global       1/1        aarnphm/bentoml-sentiment-analysis:latest   *:5000->5000/tcp
    t2heqc7is2qw   bento-prom-stack_grafana      global       1/1        grafana/grafana:latest                      *:3000->3000/tcp
    65mj931dhax6   bento-prom-stack_prometheus   replicated   1/1        prom/prometheus:latest                      *:9090->9090/tcp

View logs of specific service, eg: ``bentoml``

.. code-block:: bash

    » docker service logs bento-prom-stack_bentoml

To stop the service when we are done:

.. code-block:: bash

    » docker stop <our_service>