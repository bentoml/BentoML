========================
Packaging for deployment
========================

BentoML provides a standardized format called Bentos for packaging AI/ML services. A Bento includes all the components required to run AI services, such as source code, Python dependencies, model artifacts, and configurations. This ensures your AI services are consistent and reproducible across different environments.

Define the runtime environment
------------------------------

Before building a Bento, you need to define the runtime environment of it. Here's an example:

.. code-block:: python
    :caption: `service.py`

    import bentoml

    my_image = bentoml.images.PythonImage(python_version="3.11") \
        .python_packages("torch", "transformers")

    @bentoml.service(
        image=my_image,
        envs=[
            {"name": "HF_TOKEN", "value": "your_hf_token"},
            {"name": "DB_HOST", "value": "localhost"}
        ]
    )
    class Summarization:
        ...

Key environment fields:

- ``python_version``: Specifies the Python version to use. It defaults to the Python version in your build environment.
- ``python_packages``: Lists the required Python packages. Alternatively, reference a requirements file using ``.requirements_file("requirements.txt")``.

In the ``@bentoml.service`` decorator, apply the runtime environment to your Service via ``image``. Optionally, use the ``envs`` parameter to specify required environment variables.

See more :doc:`available fields </build-with-bentoml/runtime-environment>` to customize your build.

Build a Bento
-------------

Run the following command in the same directory as your ``service.py`` file. Replace ``<service_name>`` with the name of your Service (for example, ``Summarization`` from the earlier example).

.. code-block:: bash

   bentoml build service:<service_name>

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
