========================
Packaging for deployment
========================

BentoML provides a standardized format called Bentos for packaging AI/ML services. A Bento includes all the components required to run AI services, such as source code, Python dependencies, model artifacts, and configurations. This ensures your AI services are consistent and reproducible across different environments.

Common build options
--------------------

Build options refer to a set of configurations for building a BentoML project into a Bento. These options can be defined in a ``pyproject.toml`` file under the ``[tool.bentoml.build]`` section or a YAML file (typically named ``bentofile.yaml``).

Here's an example ``bentofile.yaml`` file for :doc:`/get-started/hello-world`.

.. code-block:: yaml

    service: "service.py:Summarization"
    include:
      - "*.py"
    python:
      packages:
        - torch
        - transformers

Key fields:

- ``service`` (Required): Points to your ``service.py`` file and the Service class defined.
- ``include``: Includes specific files in the Bento. It supports wildcard characters (for example, ``"*.py”`` and ``"path/to/file.csv"``).
- ``python.packages``: Lists required Python packages. Alternatively, reference a separate `requirements.txt <https://pip.pypa.io/en/stable/reference/requirements-file-format/>`_ file:

  .. code-block:: bash

     python:
       requirements_txt: "./requirements.txt"

For more information on available fields, see :doc:`/reference/bentoml/bento-build-options`.

Build a Bento
-------------

To build a Bento, run the following command in the same directory as your ``bentofile.yaml`` file:

.. code-block:: bash

   bentoml build

After building, each Bento is automatically assigned a unique version. You can list all available Bentos using:

.. code-block:: bash

   bentoml list

The ``bentoml build`` command is part of the ``bentoml deploy`` workflow. You should use this command only if you want to build a Bento without deploying it to BentoCloud. To deploy your project to BentoCloud directly, use ``bentoml deploy``. For details, see :doc:`cloud-deployment`.

Containerize a Bento
--------------------

To containerize a Bento with Docker, simply run ``bentoml containerize <bento_tag>``. For example:

.. code-block:: bash

    bentoml containerize summarization:latest

.. note::

    For Mac computers with Apple silicon, you can specify the ``--platform`` option to avoid potential compatibility issues with some Python libraries.

    .. code-block:: bash

        bentoml containerize --platform=linux/amd64 summarization:latest

The Docker image's tag is the same as the Bento tag by default. View the created Docker image:

.. code-block:: bash

    $ docker images

    REPOSITORY      TAG                IMAGE ID       CREATED         SIZE
    summarization   lkpxx2u5o24wpxjr   79a06b402644   2 minutes ago   6.66GB

Run the Docker image locally:

.. code-block:: bash

    docker run -it --rm -p 3000:3000 summarization:lkpxx2u5o24wpxjr serve

With the Docker image, you can run the model in any Docker-compatible environment.
