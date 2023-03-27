===============
Building Bentos
===============

What is a Bento?
----------------

:ref:`Bento 🍱 <reference/core:bentoml.Bento>` is a file archive with all the source
code, models, data files and dependency configurations required for running a
user-defined :ref:`reference/core:bentoml.Service`, packaged into a standardized format.

While ``bentoml.Service`` standardizes the inference API definition, including the
serving logic, runners initialization and API input, output types.
``Bento`` standardizes how to reproduce the required environment for running a
``bentoml.Service`` in production.

.. note::
    "Bento Build" is essentially the build process in traditional software development,
    where source code files were converted into standalone artifacts that are ready to
    deploy. BentoML reimagined this process for Machine Learning model delivery, and
    optimized the workflow both for interactive model development and for working with
    automated training pipelines.


The Build Command
-----------------

A Bento can be created with the :ref:`bentoml build <reference/cli:build>` CLI command
with a ``bentofile.yaml`` build file. Here's an example from the
:doc:`tutorial </tutorial>`:

.. code-block:: yaml

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

.. code-block:: bash

    » bentoml build

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

Similar to :doc:`saving a model </concepts/model>`, a unique version tag will be
automatically generated for the newly created Bento.

It is also possible to customize the Bento version string by specifying it in the
:code:`--version` CLI argument. However this is generally not recommended. Only use it
if your team has a very specific naming convention for deployable artifacts, e.g.:

.. code-block:: bash

    » bentoml build --version 1.0.1

.. note::

    The Bento build process requires importing the ``bentoml.Service`` object
    defined. This means, the build environment must have all its dependencies installed.
    Support for building from a docker environment is on the roadmap, see :issue:`2495`.

Advanced Project Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^

For projects that are part of a larger codebase and interacts with other local python
modules; Or for projects containing multiple Bentos/Services, it may not be possible to
put all service definition code and ``bentofile.yaml`` under the project's root
directory.

BentoML allows placing the service definition file and bentofile anywhere in the project
directory. In this case, the user needs to provide the ``build_ctx`` and
``bentofile`` argument to the ``bentoml build`` CLI command.

build_ctx
    Build context is your Python project's working directory. This is from where you
    start the Python interpreter during development so that your local python modules
    can be imported properly. Default to current directory where the
    ``bentoml build`` takes place.

bentofile
    ``bentofile`` is a ``.yaml`` file that specifies the
    :ref:`concepts/bento:Bento Build Options`. Default to the ``bentofile.yaml``
    file under the build context.

They can also be customized via the CLI command, e.g.:

.. code-block:: bash

    » bentoml build -f ./src/my_project_a/bento_fraud_detect.yaml ./src/


Managing Bentos
---------------

Bentos are the unit of deployment in BentoML, one of the most important artifact to keep
track of for your model deployment workflow.

Local Bento Store
^^^^^^^^^^^^^^^^^

Similar to Models, Bentos built locally can be managed via the
:doc:`bentoml CLI commands </reference/cli>`:

.. tab-set::

    .. tab-item:: List

       .. code-block:: bash

          » bentoml list

          Tag                               Size        Creation Time        Path
          iris_classifier:nvjtj7wwfgsafuqj  16.99 KiB   2022-05-17 21:36:36  ~/bentoml/bentos/iris_classifier/nvjtj7wwfgsafuqj
          iris_classifier:jxcnbhfv6w6kvuqj  19.68 KiB   2022-04-06 22:02:52  ~/bentoml/bentos/iris_classifier/jxcnbhfv6w6kvuqj

    .. tab-item:: Get

       .. code-block:: bash

          » bentoml get iris_classifier:latest

          service: service:svc
          name: iris_classifier
          version: nvjtj7wwfgsafuqj
          bentoml_version: 1.0.0
          creation_time: '2022-05-17T21:36:36.436878+00:00'
          labels:
            owner: bentoml-team
            project: gallery
          models:
          - tag: iris_clf:nb5vrfgwfgtjruqj
            module: bentoml.sklearn
            creation_time: '2022-05-17T21:36:27.656424+00:00'
          runners:
          - name: iris_clf
            runnable_type: SklearnRunnable
            models:
            - iris_clf:nb5vrfgwfgtjruqj
            resource_config:
              cpu: 4.0
              nvidia_gpu: 0.0
          apis:
          - name: classify
            input_type: NumpyNdarray
            output_type: NumpyNdarray


    .. tab-item:: Delete

       .. code-block:: bash

          » bentoml delete iris_classifier:latest -y

          Bento(tag="iris_classifier:nvjtj7wwfgsafuqj") deleted


Import and Export
^^^^^^^^^^^^^^^^^

Bentos can be exported to a standalone archive file outside of the store, for sharing
Bentos between teams or moving between different deployment stages. For example:

.. code:: bash

    > bentoml export iris_classifier:latest .

    INFO [cli] Bento(tag="iris_classifier:nvjtj7wwfgsafuqj") exported to ./iris_classifier-nvjtj7wwfgsafuqj.bento

.. code:: bash

    > bentoml import ./iris_classifier-nvjtj7wwfgsafuqj.bento

    INFO [cli] Bento(tag="iris_classifier:nvjtj7wwfgsafuqj") imported

.. note::

    Bentos can be exported to or import from AWS S3, GCS, FTP, Dropbox, etc. For
    example with S3:

    .. code:: bash

        pip install fs-s3fs  # Additional dependency required for working with s3
        bentoml import s3://bentoml.com/quickstart/iris_classifier.bento
        bentoml export iris_classifier:latest s3://my_bucket/my_prefix/

    To see a list of plugins usable for upload, see
    `the list <https://www.pyfilesystem.org/page/index-of-filesystems/>`_ provided by the
    pyfilesystem library.


Push and Pull
^^^^^^^^^^^^^

`Yatai <https://github.com/bentoml/Yatai>`_ provides a centralized Bento repository
that comes with flexible APIs and Web UI for managing all Bentos created by your team.
It can be configured to store Bento files on cloud blob storage such as AWS S3, MinIO
or GCS, and automatically build docker images when a new Bento was pushed.

.. code-block:: bash

  » bentoml push iris_classifier:latest

  Successfully pushed Bento "iris_classifier:nvjtj7wwfgsafuqj"

.. code-block:: bash

  » bentoml pull iris_classifier:nvjtj7wwfgsafuqj

  Successfully pulled Bento "iris_classifier:nvjtj7wwfgsafuqj"

.. image:: /_static/img/yatai-bento-repos.png
   :alt: Yatai Bento Repo UI


Bento Management API
^^^^^^^^^^^^^^^^^^^^

Similar to :ref:`concepts/model:Managing Models`, equivalent Python APIs are also
provided for managing Bentos:

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

            Bentos can be exported to or import from AWS S3, GCS, FTP, Dropbox, etc. For
            example: :code:`bentoml.export_bento('my_bento:latest', 's3://my_bucket/folder')`

    .. tab-item:: Push / Pull

        If your team has `Yatai <https://github.com/bentoml/Yatai>`_ setup, you can also
        push local Bentos to Yatai, it provides APIs and Web UI for managing all Bentos
        created by your team, stores Bento files on cloud blob storage such as AWS S3, MinIO
        or GCS, and automatically builds docker images when a new Bento was pushed.

        .. code-block:: bash

            import bentoml
            bentoml.push("iris_classifier:nvjtj7wwfgsafuqj")

        .. code-block:: bash

            bentoml.pull("iris_classifier:nvjtj7wwfgsafuqj")

    .. tab-item:: Delete

        .. code-block:: bash

            import bentoml
            bentoml.delete("iris_classifier:nvjtj7wwfgsafuqj")


What's inside a Bento
^^^^^^^^^^^^^^^^^^^^^

It is possible to view the generated files in a specific Bento. Simply use the
:code:`-o/--output` option of the ``bentoml get`` command to find the file path to
the Bento archive directory.

.. code-block:: bash

    » cd $(bentoml get iris_classifier:latest -o path)
    » tree
    .
    ├── README.md
    ├── apis
    │   └── openapi.yaml
    ├── bento.yaml
    ├── env
    │   ├── docker
    │   │   ├── Dockerfile
    │   │   └── entrypoint.sh
    │   └── python
    │       ├── requirements.lock.txt
    │       ├── requirements.txt
    │       └── version.txt
    ├── models
    │    └── iris_clf
    │       ├── latest
    │       └── nb5vrfgwfgtjruqj
    │           ├── model.yaml
    │           └── saved_model.pkl
    └── src
        ├── locustfile.py
        ├── service.py
        └── train.py


* ``src`` directory contains files specified under the :ref:`include <concepts/bento:Files to include>` field in the ``bentofile.yaml``. These
  files are relative to user Python code's CWD (current working directory), which makes
  importing relative modules and file path inside user code possible.

* ``models`` directory contains all models required by the Service. This is automatically determined from the ``bentoml.Service`` object's runners list.

* ``apis`` directory contains all API definitions. This directory contains API specs
  that are generated from the ``bentoml.Service`` object's API definitions.

* ``env`` directory contains all environment-related files which will help boostrap the Bento 🍱. This directory contains files that are generated
  from :ref:`concepts/bento:Bento Build Options` that is specified under ``bentofile.yaml``.

.. note::

   :bdg-warning:`Warning:` users **should never** change files in the generated Bento
   archive, unless it's for debugging purpose.


Bento Build Options
-------------------

Build options are specified in a ``.yaml`` file, which customizes the final Bento
produced.

By convention, this file is named ``bentofile.yaml``.

In this section, we will go over all the build options, including defining
dependencies, configuring files to include, and customize docker image settings.

Service
^^^^^^^

``service`` is a **required** field which specifies where the
``bentoml.Service`` object is defined. 

In the :doc:`tutorial </tutorial>`, we defined ``service: "service:svc"``, which can be
interpreted as:

- ``service`` refers to the Python module (the ``service.py`` file)
- ``svc`` refers to the ``bentoml.Service`` object created in ``service.py``, with ``svc = bentoml.Service(...)``

.. tip::

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
      │ » bentoml serve service:svc │
      │                             │
      │                             │
      └─────────────────────────────┘


Description
^^^^^^^^^^^

``description`` field allows user to customize documentation for any given Bento.

The description contents must be plain text, optionally in `Markdown <https://daringfireball.net/projects/markdown/syntax>`_ format. Description
can be specified either inline in the ``bentofile.yaml``, or via a file path to an
existing text file:

.. tab-set::

   .. tab-item:: Inline

      .. code-block:: yaml

          service: "service.py:svc"
          description: |
              ## Description For My Bento 🍱

              Use **any markdown syntax** here!

              > BentoML is awesome!
          include:
              ...

   .. tab-item:: File path

      .. code-block:: yaml

          service: "service.py:svc"
          description: "file: ./README.md"
          include:
              ...

.. tip::
    When pointing to a description file, it can be either an absolute path or a relative
    path. The file must exist on the given path upon ``bentoml build`` command run,
    and for relative file path, the current path is set to the ``build_ctx``, which
    default to the directory where ``bentoml build`` was executed from.


Labels
^^^^^^

``labels`` are key-value pairs that are attached to an object.

In BentoML, both ``Bento`` and ``Model`` can have labels attached to them. Labels are intended to
be used to specify identifying attributes of Bentos/Models that are meaningful and
relevant to users, but do not directly imply semantics to the rest of the system.

Labels can be used to organize models and Bentos in `Yatai <https://github.com/bentoml/Yatai>`_,
which also allow users to add or modify labels at any time.

.. code-block:: yaml

   labels:
     owner: bentoml-team
     stage: not-ready

Files to include
^^^^^^^^^^^^^^^^

In the example :ref:`above <concepts/bento:The Build Command>`, the :code:`*.py` includes every Python files under ``build_ctx``.
You can also include other wildcard and directory pattern matching.

.. code-block:: yaml

    ...
    include:
      - "data/"
      - "**/*.py"
      - "config/*.json"
      - "path/to/a/file.csv"


If the include field is not specified, BentoML will include all files under the ``build_ctx`` directory, besides the ones explicitly set to be excluded, as will be demonstrated in :ref:`concepts/bento:Files to exclude`.

.. seealso::

   Both ``include`` and ``exclude`` fields support `gitignore style pattern
   matching.  <https://git-scm.com/docs/gitignore#_pattern_format>`_.


Files to exclude
^^^^^^^^^^^^^^^^

If there are a lot of files under the working directory, another approach is to
only specify which files to be ignored.

``exclude`` field specifies the pathspecs (similar to ``.gitignore`` files) of files to be excluded in the final Bento build. The pathspecs are relative to
the ``build_ctx`` directory.

.. code-block:: yaml

    ...
    include:
    - "data/"
    - "**/*.py"
    exclude:
    - "tests/"
    - "secrets.key"

Users can also opt to place a ``.bentoignore`` file in the ``build_ctx``
directory. This is what a ``.bentoignore`` file would look like:

.. code-block:: bash
   :caption: .bentoignore

   __pycache__/
   *.py[cod]
   *$py.class
   .ipynb_checkpoints/
   training_data/

.. note::

    ``exclude`` is always applied after ``include``.


Python Packages
^^^^^^^^^^^^^^^

Required Python packages for a given Bento can be specified under the ``python.packages`` field.

When a package name is left without a version, BentoML will lock the package to the
version available under the current environment when running ``bentoml build``. User can also specify the
desired version, install from a custom PyPI source, or install from a GitHub repo:

.. code-block:: yaml

    python:
        packages:
        - "numpy"
        - "matplotlib==3.5.1"
        - "package>=0.2,<0.3"
        - "torchvision==0.9.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu"
        - "git+https://github.com/username/mylib.git@main"

.. note::
    There's no need to specify :code:`bentoml` as a dependency here since BentoML will
    addd the current version of BentoML to the Bento's dependency list by default. User
    can override this by specifying a different BentoML version.


To use a variant of BentoML with additional features such as gRPC, tracing exporters, pydantic
validation, specify the desired variant in the under ``python.packages`` field:

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

If you already have a
`requirements.txt <https://pip.pypa.io/en/stable/reference/requirements-file-format/>`_
file that defines python packages for your project, you may also supply a path to the
``requirements.txt`` file directly:

.. code-block:: yaml

    python:
        requirements_txt: "./project-a/ml-requirements.txt"

Pip Install Options
"""""""""""""""""""

Additional ``pip install`` arguments can also be provided.

Note that these arguments will be applied to all packages defined in ``python.packages``, as
well as the ``requirements_txt`` file, if provided.

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

    **BentoML by default will cache pip artifacts across all local image builds to speed
    up the build process**.

    If you want to force a re-download instead of using the cache, you can specify the :code:`pip_args: "--no-cache-dir"` option in your
    ``bentofile.yaml``, or use the :code:`--no-cache` option in ``bentoml containerize`` command, e.g.:

    .. code-block::

        » bentoml containerize my_bento:latest --no-cache


PyPI Package Locking
""""""""""""""""""""

By default, BentoML automatically locks all package versions, as well as all packages in
their dependency graph, to the version found in the current build environment, and
generates a :code:`requirements.lock.txt` file. This process uses
`pip-compile <https://github.com/jazzband/pip-tools>`_ under the hood.

If you have already specified a version for all packages, you can optionally disable
this behavior by setting the ``lock_packages`` field to False:

.. code-block:: yaml

    python:
        requirements_txt: "requirements.txt"
        lock_packages: false


Python Wheels
"""""""""""""

Python ``.whl`` files are also supported as a type of dependency to include in a
Bento. Simply provide a path to your ``.whl`` files under the ``wheels``` field.


.. code-block:: yaml

    python:
        wheels:
        - ./lib/my_package.whl


If the wheel is hosted on a local network without TLS, you can indicate
that the domain is safe to pip with the ``trusted_host`` field.

Python Options Table
""""""""""""""""""""

+-------------------+------------------------------------------------------------------------------------+
| Field             | Description                                                                        |
+===================+====================================================================================+
| requirements_txt  | The path to a custom requirements.txt file                                         |
+-------------------+------------------------------------------------------------------------------------+
| packages          | Packages to include in this bento                                                  |
+-------------------+------------------------------------------------------------------------------------+
| lock_packages     | Whether to lock the packages or not                                                |
+-------------------+------------------------------------------------------------------------------------+
| index_url         | Inputs for the ``--index-url`` pip argument                                        |
+-------------------+------------------------------------------------------------------------------------+
| no_index          | Whether to include the ``--no-index`` pip argument                                 |
+-------------------+------------------------------------------------------------------------------------+
| trusted_host      | List of trusted hosts used as inputs using the ``--trusted-host`` pip argument     |
+-------------------+------------------------------------------------------------------------------------+
| find_links        | List of links to find as inputs using the ``--find-links`` pip argument            |
+-------------------+------------------------------------------------------------------------------------+
| extra_index_url   | List of extra index urls as inputs using the ``≈`` pip argument                    |
+-------------------+------------------------------------------------------------------------------------+
| pip_args          | Any additional pip arguments that you would like to add when installing a package  |
+-------------------+------------------------------------------------------------------------------------+
| wheels            | List of paths to wheels to include in the bento                                    |
+-------------------+------------------------------------------------------------------------------------+


Conda Options
^^^^^^^^^^^^^

Conda dependencies can be specified under ``conda`` field. For example:

.. code-block:: yaml

    conda:
        channels:
        - default
        dependencies:
        - h2o
        pip:
        - "scikit-learn==1.2.0"

When ``channels`` filed is left unspecified, BentoML will use the community
maintained ``conda-forge`` channel as the default.

Optionally, you can export all dependencies from a preexisting conda environment to
an ``environment.yml`` file, and provide this file in your ``bentofile.yaml``
config:

Export conda environment:

.. code-block:: bash

    » conda env export > environment.yml

In your ``bentofile.yaml``:

.. code-block:: yaml

    conda:
        environment_yml: "./environment.yml"


.. note::

    Unlike Python packages, BentoML does not support locking conda packages versions
    automatically. It is recommended for users to specify a version in the config file.

.. seealso::

    When ``conda`` options are provided, BentoML will select a docker base image
    that comes with Miniconda pre-installed in the generated Dockerfile. Note that only
    the ``debian`` and ``alpine`` distro support ``conda``. Learn more at
    the :ref:`concepts/bento:Docker Options` section below.


Conda Options Table
"""""""""""""""""""

+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| Field            | Description                                                                                                                      |
+==================+==================================================================================================================================+
| environment_yml  | Path to a conda environment file to copy into the bento. If specified, this file will overwrite any additional option specified  |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| channels         | Custom conda channels to use. If not specified will use ``conda-forge``                                                          |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| dependencies     | Custom conda dependencies to include in the environment                                                                          |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| pip              | The specific ``pip`` conda dependencies to include                                                                               |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+


Docker Options
^^^^^^^^^^^^^^

BentoML makes it easy to deploy a Bento to a Docker container. This section discuss the
available options for customizing the docker image generated from a Bento.

Here's a basic Docker options configuration:

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
          - FOO=value1
          - BAR=value2

.. note::

   BentoML leverage `BuildKit <https://github.com/moby/buildkit>`_, a cache-efficient builder toolkit,
   to containerize Bentos 🍱.

   BuildKit comes with `Docker 18.09 <https://docs.docker.com/develop/develop-images/build_enhancements/>`_. This means
   if you are using Docker via Docker Desktop, BuildKit will be available by default.

   However, if you are using a standalone version of Docker, you can install
   BuildKit by following the instructions `here <https://github.com/docker/buildx#installing>`_.

OS Distros
""""""""""

The following OS distros are currently supported in BentoML:

- ``debian``: **default**, similar to Ubuntu
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

.. TODO::
    Document image supported architectures


GPU support
"""""""""""

The ``cuda_version`` field specifies the target CUDA version to install on the
the generated docker image. Currently, the following CUDA version are supported:

* ``"11.6.2"``
* ``"11.4.3"``
* ``"11.2.2"``

BentoML will also install additional packages required for given target CUDA version.

.. code-block:: yaml

    docker:
        cuda_version: "11.6.2"

If you need a different cuda version that is not currently supported in BentoML, it is
possible to install it by specifying it in the ``system_packages`` or via the
``setup_script``.

.. dropdown:: Installing custom CUDA version with conda
   :icon: code


   We will demonstrate how you can install custom cuda version via conda.

   Add the following to your ``bentofile.yaml``:

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

   Then proceed with ``bentoml build`` and ``bentoml containerize`` respectively:

   .. code-block:: bash

      » bentoml build

      » bentoml containerize <bento>:<tag>


Setup Script
""""""""""""

For advanced Docker customization, you can also use a ``setup_script`` to inject
arbitrary user provided script during the image build process. For example, with NLP
projects you can pre-download NLTK data in the image with:

In your ``bentofile.yaml``:

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

Now build a new bento and then run ``bentoml containerize MY_BENTO --progress plain`` to
view the docker image build progress. The newly built docker image will contain
pre-downloaded NLTK dataset.

.. tip::

    When working with bash scripts, it is recommended to add ``set -euxo pipefail``
    to the beginning. Especially when `set -e` is missing, the script will fail silently
    without raising an exception during ``bentoml containerize``. Learn more about
    `Bash Set builtin <https://www.gnu.org/software/bash/manual/html_node/The-Set-Builtin.html>`_.

It is also possible to provide a Python script for initializing the docker image. Here's
an example:

In ``bentofile.yaml``:

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

Setup script is always executed after the specified Python packages, conda dependencies,
and system packages are installed. Thus user can import and utilize those libraries in
their setup script for the initialization process.

Enable features for your Bento
""""""""""""""""""""""""""""""

Users can optionally pass in the ``--enable-features`` flag to ``bentoml containerize`` to
enable additional features for the generated Bento container image.

+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features``                 | Feature                                                                                                                 |
+=======================================+=========================================================================================================================+
| ``--enable-features=aws``             | adding AWS interop (currently file upload to S3)                                                                        |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=grpc``            | enable gRPC functionalities in BentoML                                                                                  |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=grpc-channelz``   | enable `gRPC Channelz <https://grpc.io/blog/a-short-introduction-to-channelz/>`_ for debugging purposes                 |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=grpc-reflection`` | enable :github:`gRPC Reflection <grpc/grpc/blob/master/doc/server-reflection.md>`                                       |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=io-image``        | adding Pillow dependencies to :ref:`Image IO descriptor <reference/api_io_descriptors:Images>`                          |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=io-json``         | adding Pydantic validation to :ref:`JSON IO descriptor <reference/api_io_descriptors:Structured Data with JSON>`        |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=io-pandas``       | adding Pandas dependencies to :ref:`PandasDataFrame descriptor <reference/api_io_descriptors:Tabular Data with Pandas>` |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=tracing-jaeger``  | enable :ref:`Jaeger Exporter <guides/tracing:Tracing>` for distributed tracing                                          |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=tracing-otlp``    | enable :ref:`OTLP Exporter <guides/tracing:Tracing>`   for distributed tracing                                          |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ``--enable-features=tracing-zipkin``  | enable :ref:`Zipkin Exporter <guides/tracing:Tracing>`  for distributed tracing                                         |
+---------------------------------------+-------------------------------------------------------------------------------------------------------------------------+

Advanced Options
""""""""""""""""

For advanced customization for generating docker images, see :doc:`/guides/containerization`:

1. :ref:`Using base image <guides/containerization:Custom Base Image>`
2. :ref:`Using dockerfile template <guides/containerization:Dockerfile Template>`

Docker Options Table
""""""""""""""""""""


+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Field               | Description                                                                                                                               |
+=====================+===========================================================================================================================================+
| distro              | The OS distribution on the Docker image, Default to ``debian``.                                                                           |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| python_version      | Specify which python to include on the Docker image [`3.7`, `3.8`, `3.9`, `3.10`]. Default to the Python version in build environment.    |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| cuda_version        | Specify the cuda version to install on the Docker image [:code:`11.6.2`].                                                                 |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| system_packages     | Declare system packages to be installed in the container.                                                                                 |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| env                 | Declare environment variables in the generated Dockerfile.                                                                                |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| setup_script        | A python or shell script that executes during docker build time.                                                                          |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| base_image          | A user-provided docker base image. This will override all other custom attributes of the image.                                           |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| dockerfile_template | Customize the generated dockerfile by providing a Jinja2 template that extends the default dockerfile.                                    |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
