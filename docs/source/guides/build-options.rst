=============
Build options
=============

A Bento is a format containing all the components - source code, Python packages, as well as model references and configuration - required for running a BentoML Service. Build options refer to a set of configurations defined in a YAML file (typically named ``bentofile.yaml``) for building a BentoML project into a Bento. The following is an example ``bentofile.yaml`` file for :doc:`/get-started/quickstart`.

.. code-block:: yaml

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

This page explains available Bento build options in ``bentofile.yaml``.

Bento build options
-------------------

``service``
^^^^^^^^^^^

``service`` is a **required** field and points to where a :doc:`Service object </guides/services>` resides. It is often defined as ``service: "service:class-name"``.

- ``service``: The Python module, namely the ``service.py`` file.
- ``class-name``: The class-based Service's name created in ``service.py``, decorated with ``@bentoml.service``. If you have multiple Services in ``service.py``, you can specify the main Service receiving user requests in ``bentofile.yaml``. Other Services will be started together with this main Service.

``description``
^^^^^^^^^^^^^^^

``description`` allows you to annotate your Bento with relevant documentation, which can be written in plain text or `Markdown <https://daringfireball.net/projects/markdown/syntax>`_ format.
You can either directly provide the description in ``bentofile.yaml`` or reference an external file through a path.

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

For descriptions sourced from an external file, you can use either an absolute or relative path. Make sure the file exists at the specified path when the ``bentoml build`` command is run. For relative paths, the reference point is the ``build_ctx``, which defaults to the directory from which ``bentoml build`` is executed.

``labels``
^^^^^^^^^^

``labels`` are key-value pairs associated with objects. In BentoML, both Bentos and models can have labels attached to them. These labels can serve various purposes, such as identifying or categorizing Bentos and models in BentoCloud. You can add or modify labels at any time.

.. code-block:: yaml

   labels:
     owner: bentoml-team
     stage: not-ready

``include``
^^^^^^^^^^^

You use the ``include`` field to include specific files when building the Bento. It supports wildcard characters and directory pattern matching. For example, setting it to ``*.py`` means every Python file under the existing ``build_ctx`` will be packaged into the Bento.

.. code-block:: yaml

    ...
    include:
      - "data/"
      - "**/*.py"
      - "config/*.json"
      - "path/to/a/file.csv"

If this field is not specified, BentoML includes all files under the ``build_ctx`` by default, excluding those explicitly set in the ``exclude`` field.

.. note::

   Both ``include`` and ``exclude`` fields support `gitignore style pattern matching <https://git-scm.com/docs/gitignore#_pattern_format>`_.

``exclude``
^^^^^^^^^^^

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
BentoML will lock the package to the version available in the current environment when building a Bento.

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

``conda``
^^^^^^^^^

Conda dependencies can be specified under the ``conda`` field. For example:

.. code-block:: yaml

    conda:
        channels:
        - default
        dependencies:
        - h2o
        pip:
        - "scikit-learn==1.2.0"

- ``channels``: Custom conda channels to use. If it is not specified, BentoML will use the community-maintained ``conda-forge`` channel as the default.
- ``dependencies``: Custom conda dependencies to include in the environment.
- ``pip``: The specific ``pip`` conda dependencies to include.

Optionally, you can export all dependencies from a pre-existing conda environment to an ``environment.yml`` file, and provide this file in your ``bentofile.yaml`` file. If it is specified, this file will overwrite any additional option specified.

To export a conda environment:

.. code-block:: bash

    conda env export > environment.yml

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
    the ``docker`` section below.

``docker``
^^^^^^^^^^

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

Build a Bento
-------------

With a ``bentofile.yaml`` file, you build a Bento by running ``bentoml build``. Note that this command is part of the ``bentoml deploy`` workflow. You should use this command only if you want to build a Bento without deploying it to BentoCloud.

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

After built, each Bento is automatically tagged with a unique version. It is also possible to set a specific version using the ``--version`` option,
but this is generally unnecessary. Only use it if your team has a very specific naming convention for deployable artifacts.

.. code-block:: bash

    bentoml build --version 1.0.1

Custom build context
^^^^^^^^^^^^^^^^^^^^

For projects that are part of a larger codebase and interact with other local Python
modules or those containing multiple Bentos/Services, it might not be possible to
put all Service definition code and ``bentofile.yaml`` in the project's root directory.

BentoML allows the placement of the Service definition and ``bentofile.yaml`` anywhere in the project directory.
In such scenarios, specify the ``build_ctx`` and ``bentofile`` arguments when running the ``bentoml build`` command.

* ``build_ctx``: The build context represents the working directory of your Python project. It will be prepended to the PYTHONPATH during build process,
  ensuring the correct import of local Python modules. By default, it's set to the current directory where the ``bentoml build`` command is executed.
* ``bentofile``: It defaults to the ``bentofile.yaml`` file in the build context.

To customize their values, use the following:

.. code-block:: bash

    bentoml build -f ./src/my_project_a/bento_fraud_detect.yaml ./src/

Structure
^^^^^^^^^

By default, all created Bentos are stored in the BentoML Bento Store, which is essentially a local directory. You can go to a specific Bento directory by running the following command:

.. code-block:: bash

    cd $(bentoml get BENTO_TAG -o path)

Inside the directory, you might see different files and sub-directories depending on the configurations in ``bentofile.yaml``. A typical Bento contains the following key sub-directories:

* ``src``: Contains files specified in the ``include`` field of ``bentofile.yaml``. These files are relative to user Python code's CWD (current working directory), which makes importing relative modules and file paths inside user code possible.
* ``apis``: Contains API definitions auto-generated from the Service's API specifications.
* ``env``: Contains environment-related files for Bento initialization. These files are generated based on the build options specified in ``bentofile.yaml``.

.. warning::

   We do not recommend you change files in a Bento directly, unless it's for debugging purposes.
