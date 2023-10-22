======
Bentos
======

A :ref:`Bento <reference/core:bentoml.Bento>` is an archive containing all essential components - source
code, models, data files, and dependency configurations - required for running a
user-defined :ref:`BentoML Service <reference/core:bentoml.Service>`. It is the standardized distribution unit in the BentoML ecosystem.

While the Service standardizes the inference API definition, including the serving logic, Runner initialization, and API input/output types,
the Bento provides a standardized approach to reproducing the required environment for running a ``bentoml.Service`` instance in production.

.. note::

    "Bento Build" is essentially the build process in traditional software development,
    where source code files are converted into standalone artifacts that are ready to
    deploy. BentoML reimagines this process for machine learning model delivery, and
    optimizes the workflow both for interactive model development and for working with
    automated training pipelines.

This page explains how to build and manage Bentos, and available configurations for customizing the Bento build process.

Build a Bento
-------------

To build a Bento, you use the :ref:`bentoml build <reference/cli:build>` CLI command with a ``bentofile.yaml`` configuration file.
Here is a simple example with basic configurations from the :doc:`/quickstarts/deploy-an-iris-classification-model-with-bentoml` quickstart.
For a complete list of available configurations, see :ref:`concepts/bento:Bento build options`.

.. code-block:: yaml
    :caption: `bentofile.yaml`

    service: "service:svc"  # Same as the argument passed to `bentoml serve`
    labels:
        owner: bentoml-team
        stage: dev
    include:
    - "*.py"  # A pattern for matching which files to include in the bento
    python:
        packages:  # Additional pip packages required by the service
        - scikit-learn
        - pandas
    models: # The model to be used for building the Bento.
    - iris_clf:latest

.. code-block:: bash

    $ bentoml build

    Building BentoML service "iris_classifier:dpijemevl6nlhlg6" from build context "/home/user/gallery/quickstart"
    Packing model "iris_clf:zy3dfgxzqkjrlgxi"
    Locking PyPI package versions..

    ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
    ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
    ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
    ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
    ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
    ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

    Successfully built Bento(tag="iris_classifier:dpijemevl6nlhlg6")

After built, each Bento is automatically tagged with a unique version. It is also possible to set a specific version using the ``--version`` option,
but this is generally unnecessary. Only use it if your team has a very specific naming convention for deployable artifacts.

.. code-block:: bash

    bentoml build --version 1.0.1

.. note::

    Building a Bento involves importing the ``bentoml.Service`` object and its dependencies.
    Make sure you have all the dependencies installed in the build environment.
    Support for building from a Docker environment is on the roadmap; see :issue:`2495` for details.

Custom build context
^^^^^^^^^^^^^^^^^^^^

For projects that are part of a larger codebase and interact with other local Python
modules or those containing multiple Bentos/Services, it might not be possible to
put all Service definition code and ``bentofile.yaml`` in the project's root directory.

BentoML allows the placement of the Service definition and ``bentofile.yaml`` anywhere in the project directory.
In such scenarios, specify the ``build_ctx`` and ``bentofile`` arguments when running the ``bentoml build`` command.

* ``build_ctx``: The build context represents the working directory of your Python project. It's the location where the Python interpreter starts,
  ensuring the correct import of local Python modules. By default, it's set to the current directory where the ``bentoml build`` command is executed.
* ``bentofile``: A YAML configuration file that specifies the :ref:`concepts/bento:Bento build options`. It defaults to the ``bentofile.yaml`` file in the build context.

To customize their values, use the following:

.. code-block:: bash

    bentoml build -f ./src/my_project_a/bento_fraud_detect.yaml ./src/

Structure
^^^^^^^^^

By default, all created Bentos are stored in the BentoML Bento Store, which is essentially a local directory. You can go to a specific Bento directory by running the following command:

.. code-block:: bash

    $ cd $(bentoml get BENTO_TAG -o path)

Inside the directory, you might see different files and sub-directories depending on the configurations in ``bentofile.yaml``. A typical Bento contains the following key sub-directories:

* ``src``: Contains files specified in the :ref:`include <concepts/bento:Files to include>` field of ``bentofile.yaml``. These
  files are relative to user Python code's CWD (current working directory), which makes importing relative modules and file paths inside user code possible.
* ``apis``: Contains API definitions auto-generated from the Service's API specifications.
* ``env``: Contains environment-related files for Bento initialization. These files are generated based on the :ref:`concepts/bento:Bento build options` specified in ``bentofile.yaml``.

.. warning::

   We do not recommend you change files in a Bento directly, unless it's for debugging purposes.

Manage Bentos
-------------

You can manage Bentos locally by using the :doc:`bentoml CLI commands </reference/cli>`.

.. tab-set::

    .. tab-item:: List

       To display all the Bentos in the local Bento Store:

       .. code-block:: bash

          $ bentoml list

          Tag                                                                                            Size        Creation Time
          iris_classifier:rnjnyjcwtgknsnry                                                               78.84 MiB   2023-09-19 11:12:27
          pt-stabilityai-stable-diffusion-xl-base-1-0-text2img:f898a3e026e802f68796b95e9702464bac78d76f  18.84 KiB   2023-09-08 12:10:08
          meta-llama-llama-2-7b-chat-hf-service:08751db2aca9bf2f7f80d2e516117a53d7450235                 35.24 KiB   2023-08-23 11:16:46

    .. tab-item:: Get

       To retrieve details of a specified Bento:

       .. code-block:: bash

          $ bentoml get iris_classifier:latest

          service: service:svc
          name: iris_classifier
          version: rnjnyjcwtgknsnry
          bentoml_version: 1.1.0
          creation_time: '2023-09-19T03:12:27.608017+00:00'
          labels:
            owner: bentoml-team
            project: dev
          models:
          - tag: iris_clf:zf2oioswtchconry
            module: bentoml.sklearn
            creation_time: '2023-09-19T03:01:10.996520+00:00'
          runners:
          - name: iris_clf
            runnable_type: SklearnRunnable
            embedded: false
            models:
            - iris_clf:zf2oioswtchconry
            resource_config:
              cpu: 4.0
              nvidia_gpu: 0.0
          apis:
          - name: classify
            input_type: NumpyNdarray
            output_type: NumpyNdarray


    .. tab-item:: Delete

       To delete a specific Bento:

       .. code-block:: bash

          $ bentoml delete iris_classifier:latest -y

          Bento(tag="iris_classifier:rnjnyjcwtgknsnry") deleted


Import and export Bentos
^^^^^^^^^^^^^^^^^^^^^^^^

You can export a Bento in the BentoML Bento Store as a standalone archive file and share it between teams or move it between different deployment stages. For example:

.. code:: bash

    $ bentoml export iris_classifier:latest .

    INFO [cli] Bento(tag="iris_classifier:rnjnyjcwtgknsnry") exported to ./iris_classifier-rnjnyjcwtgknsnry.bento.

.. code:: bash

    $ bentoml import ./iris_classifier-rnjnyjcwtgknsnry.bento

    INFO [cli] Bento(tag="iris_classifier:rnjnyjcwtgknsnry") imported

You can also export Bentos to and import Bentos from external storage devices, such as AWS S3, GCS, FTP and Dropbox. For example:

.. code:: bash

    pip install fs-s3fs  # Additional dependency required for working with S3
    bentoml import s3://bentoml.com/quickstart/iris_classifier.bento
    bentoml export iris_classifier:latest s3://my_bucket/my_prefix/

To see a comprehensive list of supported platforms, see `the PyFilesystem list <https://www.pyfilesystem.org/page/index-of-filesystems/>`_.

Test Bentos
^^^^^^^^^^^

After you build a Bento, it's essential to test it locally before containerizing it or pushing it to BentoCloud
for production deployment. Local testing ensures that the Bento behaves as expected and helps identify any potential
issues. Here are two methods to test a Bento locally.

Via BentoML CLI
"""""""""""""""

You can easily serve a Bento using the BentoML CLI. Replace ``BENTO_TAG`` with your specific Bento tag (for example, ``iris_classifier:latest``) in the following command.

.. code:: bash

    bentoml serve BENTO_TAG

Via bentoml.Server API
""""""""""""""""""""""

For those working within scripting environments or running Python-based tests where using the CLI might be
difficult, the ``bentoml.Server`` API offers a more programmatic way to serve and interact with your Bento.
It gives you detailed control over the server lifecycle, especially useful for debugging and iterative testing.

The following example uses the Bento ``iris_classifier:latest`` created in the quickstart :doc:`/quickstarts/deploy-an-iris-classification-model-with-bentoml`
to create an HTTP server. Note that ``GrpcServer`` is also available.

.. code:: python

    from bentoml import HTTPServer
    import numpy as np

    # Initialize the server with the Bento
    server = HTTPServer("iris_classifier:latest", production=True, port=3000, host='0.0.0.0')

    # Start the server (non-blocking by default)
    server.start(blocking=False)

    # Get a client to make requests to the server
    client = server.get_client()

    # Send a request using the client
    result = client.classify(np.array([[4.9, 3.0, 1.4, 0.2]]))
    print(result)

    # Stop the server to free up resources
    server.stop()

Alternatively, you can manage the server's lifecycle using a context manager. This ensures that the server is automatically stopped once you exit the ``with`` block.

.. code:: python

    from bentoml import HTTPServer
    import numpy as np

    server = HTTPServer("iris_classifier:latest", production=True, port=3000, host='0.0.0.0')

    with server.start() as client:
        result = client.classify(np.array([[4.9, 3.0, 1.4, 0.2]]))
        print(result)

Push and pull Bentos
^^^^^^^^^^^^^^^^^^^^

`BentoCloud <https://cloud.bentoml.com>`_ provides a centralized repository with flexible APIs
and a Web Console for managing all Bentos created by your team. After you :doc:`log in to BentoCloud </bentocloud/how-tos/manage-access-token>`,
use ``bentoml push`` and ``bentoml pull`` to upload your Bentos to and download them from BentoCloud:

.. code-block:: bash

  $ bentoml push iris_classifier:latest

  Successfully pushed Bento "iris_classifier:nvjtj7wwfgsafuqj"

.. code-block:: bash

  $ bentoml pull iris_classifier:nvjtj7wwfgsafuqj

  Successfully pulled Bento "iris_classifier:nvjtj7wwfgsafuqj"

After a Bento is uploaded to BentoCloud, it is stored in one of the Bento repositories on the **Bentos** page. Each Bento repository corresponds to a Bento set,
which contains different versions of a specific machine learning model.

.. image:: /_static/img/concepts/bentos/bento-repository-page-bentocloud.png

Bento management APIs
^^^^^^^^^^^^^^^^^^^^^

In addition to the CLI commands, BentoML also provides equivalent Python APIs for managing Bentos:

.. tab-set::

    .. tab-item:: Get

        .. code-block:: python

            import bentoml
            bento = bentoml.get("iris_classifier:latest")

            print(bento.tag)
            print(bento.path)
            print(bento.info.to_dict())

    .. tab-item:: List

        .. code-block:: python

            import bentoml
            bentos = bentoml.list()

    .. tab-item:: Import / Export

        .. code-block:: python

            import bentoml
            bentoml.export_bento('my_bento:latest', '/path/to/folder/my_bento.bento')

        .. code-block:: bash

            bentoml.import_bento('/path/to/folder/my_bento.bento')

        .. note::

            You can export Bentos to and import Bentos from external storage devices, such as AWS S3, GCS, FTP and Dropbox. For example:

            .. code-block:: python

                bentoml.export_bento('my_bento:latest', 's3://my_bucket/folder')

    .. tab-item:: Push / Pull

        If you :doc:`have access to BentoCloud </bentocloud/how-tos/manage-access-token>`, you can push local Bentos to
        or pull Bentos from it.

        .. code-block:: bash

            import bentoml
            bentoml.push("iris_classifier:nvjtj7wwfgsafuqj")

        .. code-block:: bash

            bentoml.pull("iris_classifier:nvjtj7wwfgsafuqj")

    .. tab-item:: Delete

        .. code-block:: bash

            import bentoml
            bentoml.delete("iris_classifier:nvjtj7wwfgsafuqj")

Bento build options
-------------------

BentoML allows you to customize the build configurations of a Bento using a YAML file, typically named ``bentofile.yaml``. The following sections list available configurations in this file,
including Service definitions, Python packages, models, and Docker settings.

Service
^^^^^^^

``service`` is a **required** field and points to where the ``bentoml.Service`` object resides. For example, it is often defined as ``service: "service:svc"``.

* ``service``: The Python module, namely the ``service.py`` file.
* ``svc``: The ``bentoml.Service`` object named ``svc`` created in ``service.py``, with ``svc = bentoml.Service(...)``.

.. note::

   This is synonymous to how the :ref:`bentoml serve <reference/cli:serve>` command specifies a ``bentoml.Service`` target.

   .. code-block:: zsh

                           ┌──────────────┐
          ┌────────────────┤bentofile.yaml│
          │                └───────────┬──┘
          │                            │
          │  service: "service:svc"    │
          │                ─┬─         │
          │                 │          │
          └─────────────────┼──────────┘
                            │
                            │
                            │    ┌────┐
      ┌─────────────────────┼────┤bash│
      │                     │    └──┬─┘
      │                     ▼       │
      │ $ bentoml serve service:svc │
      │                             │
      │                             │
      └─────────────────────────────┘


Description
^^^^^^^^^^^

``description`` allows you to annotate your Bento with relevant documentation, which can be written in plain text or `Markdown <https://daringfireball.net/projects/markdown/syntax>`_ format.
You can either directly provide the description in the ``bentofile.yaml`` file or reference an external file through a path.

.. tab-set::

   .. tab-item:: Inline

      .. code-block:: yaml

          service: "service:svc"
          description: |
              ## Description For My Bento 🍱

              Use **any markdown syntax** here!

              > BentoML is awesome!
          include:
              ...

   .. tab-item:: File path

      .. code-block:: yaml

          service: "service:svc"
          description: "file: ./README.md"
          include:
              ...

.. note::

    For descriptions sourced from an external file, either an absolute or relative path can be used.
    Make sure the file exists at the specified path when the ``bentoml build`` command is run.
    For relative paths, the reference point is the ``build_ctx``, which defaults to the directory from which ``bentoml build`` is executed.

Labels
^^^^^^

``labels`` are key-value pairs associated with objects. In BentoML, both Bentos and models can have labels attached to them.
These labels can serve various purposes, such as identifying or categorizing Bentos and models in BentoCloud. You can add or modify labels at any time.

.. code-block:: yaml

   labels:
     owner: bentoml-team
     stage: not-ready

Files to include
^^^^^^^^^^^^^^^^

You use the ``include`` field to include specific files when building the Bento. It supports wildcard characters and directory pattern matching. For example,
setting it to ``*.py`` means every Python files under the existing ``build_ctx`` will be packaged into the Bento.

.. code-block:: yaml

    ...
    include:
      - "data/"
      - "**/*.py"
      - "config/*.json"
      - "path/to/a/file.csv"

If this field is not specified, BentoML includes all files under the ``build_ctx`` by default, excluding those explicitly set in the ``exclude`` field.

.. seealso::

   Both ``include`` and ``exclude`` fields support `gitignore style pattern
   matching <https://git-scm.com/docs/gitignore#_pattern_format>`_.

Files to exclude
^^^^^^^^^^^^^^^^

You use the ``exclude`` field to exclude specific files when building the Bento. This is useful when you have many files in the working directory, as you only need to
specify the files to be ignored.

When setting this field, you specify the file pathspecs (similar to ``.gitignore``) that are relative to the ``build_ctx`` directory.

.. code-block:: yaml

    ...
    include:
    - "data/"
    - "**/*.py"
    exclude:
    - "tests/"
    - "secrets.key"

Alternatively, create a ``.bentoignore`` file in the ``build_ctx`` directory as follows:

.. code-block:: bash
   :caption: .bentoignore

   __pycache__/
   *.py[cod]
   *$py.class
   .ipynb_checkpoints/
   training_data/

.. note::

    ``exclude`` is always applied after ``include``.

Python packages
^^^^^^^^^^^^^^^

You specify the required Python packages for a given Bento using the ``python.packages`` field. BentoML allows you to specify the
desired version and install a package from a custom PyPI source or from a GitHub repository. If a package lacks a specific version,
BentoML will lock the package to the version available in the current environment when running ``bentoml build``.

.. code-block:: yaml

    python:
        packages:
        - "numpy"
        - "matplotlib==3.5.1"
        - "package>=0.2,<0.3"
        - "torchvision==0.9.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu"
        - "git+https://github.com/username/mylib.git@main"

.. note::

    You don't need to specify ``bentoml`` as a dependency in this field since the current version of BentoML will be added to the list by default. However,
    you can override this by specifying a different BentoML version.

To use a variant of BentoML with additional features such as gRPC, tracing exporters, and Pydantic
validation, specify the desired variant in the ``python.packages`` field:

.. tab-set::

   .. tab-item:: gRPC

      .. code-block:: yaml

         python:
           packages:
           - "bentoml[grpc]"

   .. tab-item:: AWS

      .. code-block:: yaml

         python:
           packages:
           - "bentoml[aws]"

   .. tab-item:: JSON IO

      .. code-block:: yaml

         python:
           packages:
           - "bentoml[io-json]"

   .. tab-item:: Image IO

      .. code-block:: yaml

         python:
           packages:
           - "bentoml[io-image]"

   .. tab-item:: Pandas IO

      .. code-block:: yaml

         python:
           packages:
           - "bentoml[io-pandas]"

   .. tab-item:: JSON IO

      .. code-block:: yaml

         python:
           packages:
           - "bentoml[io-json]"

   .. tab-item:: Jaeger

      .. code-block:: yaml

         python:
           packages:
           - "bentoml[tracing-jaeger]"

   .. tab-item:: Zipkin

      .. code-block:: yaml

         python:
           packages:
           - "bentoml[tracing-zipkin]"

   .. tab-item:: OTLP

      .. code-block:: yaml

         python:
           packages:
           - "bentoml[tracing-otlp]"

If you already have a `requirements.txt <https://pip.pypa.io/en/stable/reference/requirements-file-format/>`_
file that defines Python packages for your project, you may also supply a path to the ``requirements.txt`` file directly:

.. code-block:: yaml

    python:
        requirements_txt: "./project-a/ml-requirements.txt"

Pip install options
"""""""""""""""""""

You can provide additional ``pip install`` arguments in the ``python`` field. If provided, these arguments will be applied to all packages defined in ``python.packages`` as
well as the ``requirements_txt`` file.

.. code-block:: yaml

    python:
        requirements_txt: "./requirements.txt"
        index_url: "https://my.mirror.com/simple"
        no_index: False
        trusted_host:
        - "pypi.python.org"
        - "my.mirror.com"
        find_links:
        - "https://download.pytorch.org/whl/cu80/stable.html"
        extra_index_url:
        - "https://<other api token>:@my.mirror.com/pypi/simple"
        - "https://pypi.python.org/simple"
        pip_args: "--pre -U --force-reinstall"

.. note::

    **By default, BentoML caches pip artifacts across all local image builds to speed up the build process**.

    If you want to force a re-download instead of using the cache, you can specify the ``pip_args: "--no-cache-dir"`` option in your
    ``bentofile.yaml`` file, or use the ``--no-cache`` option in the ``bentoml containerize`` command. For example:

    .. code-block:: bash

        $ bentoml containerize my_bento:latest --no-cache

PyPI package locking
""""""""""""""""""""

By default, BentoML automatically locks all package versions, as well as all packages in
their dependency graph, to the versions found in the current build environment, and
generates a :code:`requirements.lock.txt` file. This process uses
`pip-compile <https://github.com/jazzband/pip-tools>`_ under the hood.

If you have already specified a version for all packages, you can optionally disable
this behavior by setting the ``lock_packages`` field to ``false``:

.. code-block:: yaml

    python:
        requirements_txt: "requirements.txt"
        lock_packages: false

Python wheels
"""""""""""""

Python ``.whl`` files are also supported as a type of dependency to include in a
Bento. Simply provide a path to your ``.whl`` files under the ``wheels`` field.

.. code-block:: yaml

    python:
        wheels:
        - ./lib/my_package.whl

If the wheel is hosted on a local network without TLS, you can indicate
that the domain is safe to pip with the ``trusted_host`` field.

Python options table
""""""""""""""""""""

The following table provides a full list of available configurations for the ``python`` field.

+-------------------+------------------------------------------------------------------------------------+
| Field             | Description                                                                        |
+===================+====================================================================================+
| requirements_txt  | The path to a custom ``requirements.txt`` file                                     |
+-------------------+------------------------------------------------------------------------------------+
| packages          | Packages to include in this Bento                                                  |
+-------------------+------------------------------------------------------------------------------------+
| lock_packages     | Whether to lock the packages                                                       |
+-------------------+------------------------------------------------------------------------------------+
| index_url         | Inputs for the ``--index-url`` pip argument                                        |
+-------------------+------------------------------------------------------------------------------------+
| no_index          | Whether to include the ``--no-index`` pip argument                                 |
+-------------------+------------------------------------------------------------------------------------+
| trusted_host      | List of trusted hosts used as inputs using the ``--trusted-host`` pip argument     |
+-------------------+------------------------------------------------------------------------------------+
| find_links        | List of links to find as inputs using the ``--find-links`` pip argument            |
+-------------------+------------------------------------------------------------------------------------+
| extra_index_url   | List of extra index URLs as inputs using the ``≈`` pip argument                    |
+-------------------+------------------------------------------------------------------------------------+
| pip_args          | Any additional pip arguments that you want to add when installing a package        |
+-------------------+------------------------------------------------------------------------------------+
| wheels            | List of paths to wheels to include in the Bento                                    |
+-------------------+------------------------------------------------------------------------------------+

Models
^^^^^^

You can specify the model to be used for building a Bento using a string model tag or a dictionary, which will be written to the ``bento.yaml`` file in the Bento package.
When you start from an existing project, you can download models from BentoCloud to your local Model Store with these configurations by running ``bentoml models pull``.
Note that you need to :doc:`log in to BentoCloud </bentocloud/how-tos/manage-access-token>` first.

See the following example for details. If you don't define ``models`` in ``bentofile.yaml``, the model specified in the Service is used to build the Bento.

.. code-block:: yaml

    models:
      - "iris_clf:latest" # A string model tag
      - tag: "iris_clf:version1" # A dictionary
        filter: "label:staging"
        alias: "iris_clf_v1"

- ``tag``: The name and version of the model, separated by a colon.
- ``filter``: This field uses the same filter syntax in BentoCloud. You use a filter to list specific models, such as the models with the same label. You can add multiple comma-separated filters to a model.
- ``alias``: An alias for the model. If this is specified, you can use it directly in code like ``bentoml.models.get(alias)``.

Conda options
^^^^^^^^^^^^^

Conda dependencies can be specified under the ``conda`` field. For example:

.. code-block:: yaml

    conda:
        channels:
        - default
        dependencies:
        - h2o
        pip:
        - "scikit-learn==1.2.0"

When the ``channels`` fieed is not specified, BentoML will use the community-maintained ``conda-forge`` channel as the default.

Optionally, you can export all dependencies from a pre-existing conda environment to an ``environment.yml`` file, and provide this file in your ``bentofile.yaml`` file.

To export a conda environment:

.. code-block:: bash

    $ conda env export > environment.yml

To add it in your ``bentofile.yaml``:

.. code-block:: yaml

    conda:
        environment_yml: "./environment.yml"

.. note::

    Unlike Python packages, BentoML does not support locking conda package versions
    automatically. We recommend you specify a version in the configuration file.

.. seealso::

    When ``conda`` options are provided, BentoML will select a Docker base image
    that comes with Miniconda pre-installed in the generated Dockerfile. Note that only
    the ``debian`` and ``alpine`` distro support ``conda``. Learn more in
    the :ref:`concepts/bento:Docker Options` section below.

Conda options table
"""""""""""""""""""

The following table provides a full list of available configurations for the ``conda`` field.

+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| Field            | Description                                                                                                                      |
+==================+==================================================================================================================================+
| environment_yml  | Path to a conda environment file to copy into the Bento. If specified, this file will overwrite any additional option specified  |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| channels         | Custom conda channels to use. If not specified, BentoML will use ``conda-forge``                                                 |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| dependencies     | Custom conda dependencies to include in the environment                                                                          |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| pip              | The specific ``pip`` conda dependencies to include                                                                               |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+

Docker options
^^^^^^^^^^^^^^

BentoML makes it easy to deploy a Bento to a Docker container. It provides a set of options for customizing the Docker image generated from a Bento.

The following ``docker`` field contains some basic Docker configurations:

.. code-block:: yaml

    docker:
        distro: debian
        python_version: "3.8.12"
        cuda_version: "11.6.2"
        system_packages:
          - libblas-dev
          - liblapack-dev
          - gfortran
        env:
          FOO: value1
          BAR: value2

.. note::

   BentoML uses `BuildKit <https://github.com/moby/buildkit>`_, a cache-efficient builder toolkit, to containerize Bentos.

   BuildKit comes with `Docker 18.09 <https://docs.docker.com/develop/develop-images/build_enhancements/>`_. This means
   if you are using Docker via Docker Desktop, BuildKit will be available by default. If you are using a standalone version of Docker,
   you can install BuildKit by following the instructions `here <https://github.com/docker/buildx#installing>`_.

The following sections provide detailed explanations of available Docker configurations.

OS distros
""""""""""

The following OS distros are currently supported in BentoML:

- ``debian``: The **default** value, similar to Ubuntu
- ``alpine``: A minimal Docker image based on Alpine Linux
- ``ubi8``: Red Hat Universal Base Image
- ``amazonlinux``: Amazon Linux 2

Some of the distros may not support using conda or specifying CUDA for GPU. Here is the
support matrix for all distros:

+------------------+-----------------------------+-----------------+----------------------+
| Distro           |  Available Python Versions  | Conda Support   | CUDA Support (GPU)   |
+==================+=============================+=================+======================+
| debian           |  3.7, 3.8, 3.9, 3.10        |  Yes            |  Yes                 |
+------------------+-----------------------------+-----------------+----------------------+
| alpine           |  3.7, 3.8, 3.9, 3.10        |  Yes            |  No                  |
+------------------+-----------------------------+-----------------+----------------------+
| ubi8             |  3.8, 3.9                   |  No             |  Yes                 |
+------------------+-----------------------------+-----------------+----------------------+
| amazonlinux      |  3.7, 3.8                   |  No             |  No                  |
+------------------+-----------------------------+-----------------+----------------------+

GPU support
"""""""""""

You use the ``cuda_version`` field to specify the target CUDA version to install on the
the generated Docker image. Currently, the following CUDA versions are supported:

* ``"11.6.2"``
* ``"11.4.3"``
* ``"11.2.2"``

BentoML will also install additional packages required for a given target CUDA version.

.. code-block:: yaml

    docker:
        cuda_version: "11.6.2"

If you need a different CUDA version that is not currently supported in BentoML, you can install it by specifying it in the ``system_packages`` or the
``setup_script`` field.

.. dropdown:: Install a custom CUDA version with conda
   :icon: code

   Do the following to install a custom CUDA version via conda.

   1. Add the following to your ``bentofile.yaml`` file:

      .. code-block:: yaml

         conda:
           channels:
           - conda-forge
           - nvidia
           - defaults
           dependencies:
           - cudatoolkit-dev=10.1
           - cudnn=7.6.4
           - cxx-compiler=1.0
           - mpi4py=3.0 # installs cuda-aware openmpi
           - matplotlib=3.2
           - networkx=2.4
           - numba=0.48
           - pandas=1.0

   2. Proceed with ``bentoml build`` and ``bentoml containerize`` respectively:

      .. code-block:: bash

         $ bentoml build

         $ bentoml containerize <bento>:<tag>

Setup script
""""""""""""

For advanced Docker customization, you can also use the ``setup_script`` field to inject
any script during the image build process. For example, with NLP
projects you can pre-download NLTK data in the image by setting the following values.

In the ``bentofile.yaml`` file:

.. code-block:: yaml

    ...
    python:
      packages:
        - nltk
    docker:
      setup_script: "./setup.sh"

In the ``setup.sh`` file:

.. code-block:: bash

    #!/bin/bash
    set -euxo pipefail

    echo "Downloading NLTK data.."
    python -m nltk.downloader all

Build a new Bento and then run ``bentoml containerize MY_BENTO --progress plain`` to
view the Docker image build progress. The newly built Docker image will contain the
pre-downloaded NLTK dataset.

.. tip::

    When working with bash scripts, we recommend you add ``set -euxo pipefail``
    to the beginning. Especially when `set -e` is missing, the script will fail silently
    without raising an exception during ``bentoml containerize``. Learn more about
    `Bash Set builtin <https://www.gnu.org/software/bash/manual/html_node/The-Set-Builtin.html>`_.

It is also possible to provide a Python script for initializing the Docker image. Here's
an example:

In the ``bentofile.yaml`` file:

.. code-block:: yaml

    ...
    python:
      packages:
          - nltk
    docker:
      setup_script: "./setup.py"

In the ``setup.py`` file:

.. code-block:: python

    #!/usr/bin/env python

    import nltk

    print("Downloading NLTK data..")
    nltk.download('treebank')

.. note::

    Pay attention to ``#!/bin/bash`` and ``#!/usr/bin/env python`` in the
    first line of the example scripts above. They are known as `Shebang <https://en.wikipedia.org/wiki/Shebang_(Unix)>`_
    and they are required in a setup script provided to BentoML.

Setup scripts are always executed after the specified Python packages, conda dependencies,
and system packages are installed. Therefore, you can import and utilize those libraries in
your setup script for the initialization process.

Enable features for your Bento
""""""""""""""""""""""""""""""

You can optionally pass in the ``--enable-features`` flag to ``bentoml containerize`` to
enable additional features for the generated Bento image.

+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features``                 | Feature                                                                                                                 |
+=======================================+=========================================================================================================================+
| ``--enable-features=aws``             | Add AWS interop (currently file upload to S3)                                                                           |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=grpc``            | Enable gRPC functionalities in BentoML                                                                                  |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=grpc-channelz``   | Enable `gRPC Channelz <https://grpc.io/blog/a-short-introduction-to-channelz/>`_ for debugging purposes                 |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=grpc-reflection`` | Enable :github:`gRPC Reflection <grpc/grpc/blob/master/doc/server-reflection.md>`                                       |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=io-image``        | Add Pillow dependencies to :ref:`Image IO descriptor <reference/api_io_descriptors:Images>`                             |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=io-json``         | Add Pydantic validation to :ref:`JSON IO descriptor <reference/api_io_descriptors:Structured Data with JSON>`           |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=io-pandas``       | Add Pandas dependencies to :ref:`PandasDataFrame descriptor <reference/api_io_descriptors:Tabular Data with Pandas>`    |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=tracing-jaeger``  | Enable :ref:`Jaeger Exporter <guides/tracing:Tracing>` for distributed tracing                                          |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=tracing-otlp``    | Enable :ref:`OTLP Exporter <guides/tracing:Tracing>`   for distributed tracing                                          |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=tracing-zipkin``  | Enable :ref:`Zipkin Exporter <guides/tracing:Tracing>`  for distributed tracing                                         |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=monitor-otlp``    | Enable :ref:`Monitoring feature <guides/monitoring:Inference Data Collection & Model Monitoring>`                       |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+

Advanced options
""""""""""""""""

For advanced customization for generating Docker images, see :doc:`/guides/containerization`:

1. :ref:`Using base image <guides/containerization:Custom Base Image>`
2. :ref:`Using Dockerfile template <guides/containerization:Dockerfile Template>`

Docker options table
""""""""""""""""""""

The following table provides a full list of available configurations for the ``docker`` field.

+---------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| Field               | Description                                                                                                                                    |
+=====================+================================================================================================================================================+
| distro              | The OS distribution on the Docker image. It defaults to ``debian``.                                                                            |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| python_version      | The Python version on the Docker image [``3.7``, ``3.8``, ``3.9``, ``3.10``]. It defaults to the Python version in the build environment.      |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| cuda_version        | The CUDA version on the Docker image [``11.6.2``].                                                                                             |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| system_packages     | The system packages that will be installed in the container.                                                                                   |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| env                 | The environment variables in the generated Dockerfile.                                                                                         |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| setup_script        | A Python or Shell script that will be executed during the Docker build process.                                                                |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| base_image          | A user-provided Docker base image. This will override all other custom attributes of the image.                                                |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
| dockerfile_template | Customize the generated Dockerfile by providing a Jinja2 template that extends the default Dockerfile.                                         |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------------------+
