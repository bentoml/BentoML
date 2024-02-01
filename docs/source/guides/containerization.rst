================
Containerization
================

After defining and testing your BentoML :doc:`Service </guides/services>`, you can deploy it as an OCI-compliant image.

Prerequisites
-------------

Make sure you have `installed Docker <https://docs.docker.com/engine/install/>`_.

Build a Bento
-------------

The first step is to package your entire project into the standard distribution format in BentoML, or a Bento. To build a Bento, you need a configuration YAML file (by convention, it's ``bentofile.yaml``). This file defines the build options, such as dependencies and Docker image settings. When a Bento is being created, BentoML creates a Dockerfile within the Bento automatically.

The example file below lists the basic information required to build a Bento for :doc:`/get-started/quickstart`.

.. code-block:: yaml
    :caption: `bentofile.yaml`

    service: 'service:Summarization'
    labels:
      owner: bentoml-team
      project: gallery
    include:
      - '*.py'
    python:
      packages:
        - torch
        - transformers

Run ``bentoml build`` in your project directory to build the Bento. All created Bentos are stored in ``/home/user/bentoml/bentos/`` by default.

.. code-block:: bash

    $ bentoml build

    Locking PyPI package versions.

    ██████╗ ███████╗███╗   ██╗████████╗ ██████╗ ███╗   ███╗██╗
    ██╔══██╗██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗████╗ ████║██║
    ██████╔╝█████╗  ██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║██║
    ██╔══██╗██╔══╝  ██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║██║
    ██████╔╝███████╗██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║███████╗
    ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚══════╝

    Successfully built Bento(tag="summarization:lkpxx2u5o24wpxjr").

    Possible next steps:

     * Containerize your Bento with `bentoml containerize`:
        $ bentoml containerize summarization:lkpxx2u5o24wpxjr  [or bentoml build --containerize]

     * Push to BentoCloud with `bentoml push`:
        $ bentoml push summarization:lkpxx2u5o24wpxjr [or bentoml build --push]

View all available Bentos:

.. code-block:: bash

    $ bentoml list

    Tag                                     Size       Model Size  Creation Time
    summarization:lkpxx2u5o24wpxjr          17.08 KiB  0.00 B      2024-01-15 12:36:44

Deploy the Bento
----------------

To containerize the Bento with Docker, simply run:

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

If you prefer a serverless platform to build and operate AI applications, you can deploy Bentos to BentoCloud. It gives AI application developers a collaborative environment and a user-friendly toolkit to ship and iterate AI products.
