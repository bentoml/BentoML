===================
Bento build options
===================

Build options refer to a set of configurations for building a BentoML project into a Bento. These options can be defined in a ``pyproject.toml`` file under the ``[tool.bentoml.build]`` section or a YAML file (typically named ``bentofile.yaml``).

Available fields
----------------

``service``
^^^^^^^^^^^

``service`` is a **required** field and points to where a :doc:`Service object </build-with-bentoml/services>` resides. It is often defined as ``service: "service:class-name"``.

- ``service``: The Python module, namely the ``service.py`` file.
- ``class-name``: The class-based Service's name created in ``service.py``, decorated with ``@bentoml.service``. If you have multiple Services in ``service.py``, you can specify the main Service receiving user requests in the configuration. Other Services will be started together with this main Service.

.. tab-set::

    .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build]
         service = "service:MyService"

    .. tab-item:: bentofile.yaml

      .. code-block:: yaml

         service: "service:MyService"

``description``
^^^^^^^^^^^^^^^

``description`` allows you to annotate your Bento with relevant documentation, which can be written in plain text or `Markdown <https://daringfireball.net/projects/markdown/syntax>`_ format. You can either provide the description inline or reference an external file.

.. tab-set::

    .. tab-item:: pyproject.toml

      To define the description inline:

      .. code-block:: toml

         [tool.bentoml.build]
         service = "service:svc"
         description = "This is an inline description for the Service. BentoML is awesome!"

      To define the description using a file:

      .. code-block:: toml

         [tool.bentoml.build]
         service = "service:svc"
         description = "file: ./README.md"

    .. tab-item:: bentofile.yaml

      To define the description inline:

      .. code-block:: yaml

          service: "service:svc"
          description: |
              ## Description For My Bento 🍱

              Use **any markdown syntax** here!

              > BentoML is awesome!

      To define the description using a file:

      .. code-block:: yaml

         service: "service:svc"
         description: "file: ./README.md"

For descriptions sourced from an external file, you can use either an absolute or relative path. Make sure the file exists at the specified path when the ``bentoml build`` command is run. For relative paths, the reference point is the ``build_ctx``, which defaults to the directory from which ``bentoml build`` is executed.

``labels``
^^^^^^^^^^

``labels`` are key-value pairs associated with objects. In BentoML, both Bentos and models can have labels attached to them. These labels can serve various purposes, such as identifying or categorizing Bentos and models in BentoCloud. You can add or modify labels at any time.

.. tab-set::

    .. tab-item:: pyproject.toml

       .. code-block:: toml

         [tool.bentoml.build.labels]
         owner = "bentoml-team"
         stage = "not-ready"

    .. tab-item:: bentofile.yaml

       .. code-block:: yaml

          labels:
            owner: bentoml-team
            stage: not-ready

``include``
^^^^^^^^^^^

You use the ``include`` field to include specific files when building the Bento. It supports wildcard characters and directory pattern matching. For example, setting it to ``*.py`` means every Python file under the existing ``build_ctx`` will be packaged into the Bento.

.. tab-set::

    .. tab-item:: pyproject.toml

       .. code-block:: toml

         [tool.bentoml.build]
         include = ["data/", "**/*.py", "config/*.json", "path/to/a/file.csv"]

    .. tab-item:: bentofile.yaml

       .. code-block:: yaml

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

.. tab-set::

    .. tab-item:: pyproject.toml

       .. code-block:: toml

         [tool.bentoml.build]
         include = ["data/", "**/*.py"]
         exclude = ["tests/", "secrets.key"]

    .. tab-item:: bentofile.yaml

       .. code-block:: yaml

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

.. _build-options-model:

``models``
^^^^^^^^^^

You can specify the model to be used for building a Bento using a string model tag or a dictionary. If you don't define models in the configuration, the model specified in the Service is used to build the Bento.

.. tab-set::

    .. tab-item:: pyproject.toml

       .. code-block:: toml

          [tool.bentoml.build]
          models = [
            "summarization-model:latest",
            { tag = "summarization-model:version1", filter = "label:staging", alias = "summarization-model_v1" }
          ]

    .. tab-item:: bentofile.yaml

       .. code-block:: yaml

          models:
            - "summarization-model:latest" # A string model tag
            - tag: "summarization-model:version1"  # A dictionary
              filter: "label:staging"
              alias: "summarization-model_v1"

- ``tag``: The name and version of the model, separated by a colon.
- ``filter``: This field uses the same filter syntax in BentoCloud. You use a filter to list specific models, such as the models with the same label. You can add multiple comma-separated filters to a model.
- ``alias``: An alias for the model. If this is specified, you can use it directly in code like ``bentoml.models.get(alias)``.

Python packages
^^^^^^^^^^^^^^^

You specify the required Python packages for a given Bento using the ``python.packages`` field. BentoML allows you to specify the
desired version and install a package from a custom PyPI source or from a GitHub repository. If a package lacks a specific version,
BentoML will lock the versions of all Python packages for the current platform and Python when building a Bento.

.. tab-set::

    .. tab-item:: pyproject.toml

       When using a ``pyproject.toml`` file, you can define the required Python packages through ``project.dependencies``:

       .. code-block:: toml

          [project]
          dependencies = [
            "numpy",
            "matplotlib==3.5.1",
            "package>=0.2,<0.3",
            "torchvision==0.9.2",
            "git+https://github.com/username/mylib.git@main",
          ]

       This is equivalent to using ``python.packages``:

       .. code-block:: toml

          [tool.bentoml.build.python]
          packages = [
            "numpy",
            "matplotlib==3.5.1",
            "package>=0.2,<0.3",
            "torchvision==0.9.2",
            "git+https://github.com/username/mylib.git@main"
          ]

       If you specify both ``project.dependencies`` and ``tool.bentoml.build.python.packages``, the dependencies are combined together.

    .. tab-item:: bentofile.yaml

       .. code-block:: yaml

          python:
            packages:
              - "numpy"
              - "matplotlib==3.5.1"
              - "package>=0.2,<0.3"
              - "torchvision==0.9.2"
              - "git+https://github.com/username/mylib.git@main"

.. note::

    You don't need to specify ``bentoml`` as a dependency in this field since the current version of BentoML will be added to the list by default. However,
    you can override this by specifying a different BentoML version.

To include a package from a GitHub repository, use the `pip requirements file format <https://pip.pypa.io/en/stable/reference/requirements-file-format/>`_. You can specify the repository URL, the branch, tag, or commit to install from, and the subdirectory if the Python package is not in the root of the repository.

.. tab-set::

   .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build.python]
         packages = [
             "git+https://github.com/username/repository.git@branch_name",
             "git+https://github.com/username/repository.git@v1.0.0",
             "git+https://github.com/username/repository.git@abcdef1234567890abcdef1234567890abcdef12",
             "git+https://github.com/username/repository.git@branch_name#subdirectory=package_dir",
         ]

   .. tab-item:: bentofile.yaml

      .. code-block:: yaml

         python:
           packages:
             - "git+https://github.com/username/repository.git@branch_name"
             - "git+https://github.com/username/repository.git@v1.0.0"
             - "git+https://github.com/username/repository.git@abcdef1234567890abcdef1234567890abcdef12"
             - "git+https://github.com/username/repository.git@branch_name#subdirectory=package_dir"

If your project depends on a private GitHub repository, you can include the Python package from the repository via SSH. Make sure that the environment where BentoML is running has the appropriate SSH keys configured and that `these keys are added to GitHub <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_. In the following example, ``git@github.com:username/repository.git`` is the SSH URL for the repository.

.. tab-set::

   .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build.python]
         packages = [
             "git+ssh://git@github.com/username/repository.git@branch_name"
         ]

   .. tab-item:: bentofile.yaml

      .. code-block:: yaml

          python:
            packages:
              - "git+ssh://git@github.com/username/repository.git@branch_name"

If you already have a `requirements.txt <https://pip.pypa.io/en/stable/reference/requirements-file-format/>`_
file that defines Python packages for your project, you may also supply a path to the ``requirements.txt`` file directly:

.. tab-set::

   .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build.python]
         requirements_txt = "./project-a/ml-requirements.txt"

   .. tab-item:: bentofile.yaml

      .. code-block:: yaml

         python:
           requirements_txt: "./project-a/ml-requirements.txt"

Pip install options
"""""""""""""""""""

You can provide additional ``pip install`` arguments in the ``python`` field. If provided, these arguments will be applied to all packages defined in ``python.packages`` as
well as the ``requirements_txt`` file.

.. tab-set::

   .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build.python]
         requirements_txt = "./requirements.txt"
         index_url = "https://my.mirror.com/simple"
         no_index = false
         trusted_host = ["pypi.python.org", "my.mirror.com"]
         find_links = ["https://download.pytorch.org/whl/cu80/stable.html"]
         extra_index_url = [
             "https://<other api token>:@my.mirror.com/pypi/simple",
             "https://pypi.python.org/simple"
         ]
         pip_args = "--pre -U --force-reinstall"

   .. tab-item:: bentofile.yaml

      .. code-block:: yaml

         python:
           requirements_txt: "./requirements.txt"
           index_url: "https://my.mirror.com/simple"
           no_index: false
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
their dependency graph, and
generates a :code:`requirements.lock.txt` file. This process uses
`pip-compile <https://github.com/jazzband/pip-tools>`_ under the hood.

If you have already specified a version for all packages, you can optionally disable
this behavior by setting the ``lock_packages`` field to ``false``:

.. tab-set::

   .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build.python]
         requirements_txt = "./requirements.txt"
         lock_packages = false

   .. tab-item:: bentofile.yaml

      .. code-block:: yaml

         python:
           requirements_txt: "./requirements.txt"
           lock_packages: false

When including Python packages from GitHub repositories, use the ``pack_git_packages`` option (it defaults to ``true``) to control whether these packages should be cloned and packaged during the build process. This is useful for dependencies that may not be available via standard PyPI sources or for ensuring consistency with specific versions (for example, tags and commits) of a dependency directly from a Git repository.

.. tab-set::

   .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build.python]
         pack_git_packages = true
         packages = ["git+https://github.com/username/repository.git@abcdef1234567890abcdef1234567890abcdef12"]

   .. tab-item:: bentofile.yaml

      .. code-block:: yaml

         python:
           pack_git_packages: true  # Enable packaging of Git-based packages
           packages:
             - "git+https://github.com/username/repository.git@abcdef1234567890abcdef1234567890abcdef12"

Note that ``lock_packages`` controls whether the versions of all dependencies, not just those from Git, are pinned at the time of building the Bento. Disabling ``pack_git_packages`` will also disable package locking (``lock_packages``) unless explicitly set.

.. note::

  BentoML will always try to lock the package versions against Linux x86_64 platform to match the deployment target. If the bento contains dependencies or transitive dependencies with environment markers, they will be resolved against Linux x86_64 platform.

  For example, if the bento requires ``torch``, ``nvidia-*`` packages will also be picked up into the final lock result although they are only required for Linux x86_64 platform.

  If you want to build a bento for a different platform, you can pass ``--platform`` option to ``bentoml build`` command with the name of the target platform. For example:

  .. code-block:: bash

    $ bentoml build --platform macos

Python wheels
"""""""""""""

Python ``.whl`` files are also supported as a type of dependency to include in a
Bento. Simply provide a path to your ``.whl`` files under the ``wheels`` field.

.. tab-set::

   .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build.python]
         wheels = ["./lib/my_package.whl"]

   .. tab-item:: bentofile.yaml

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

``envs``
^^^^^^^^

Environment variables are important for managing configuration and secrets in a secure and flexible manner. They allow you to configure BentoML Services without hard-coding sensitive information, such as API keys, database credentials, or configurable parameters that might change between different environments.

You set environment variables under the ``envs`` key in ``bentofile.yaml``. Each environment variable is defined with ``name`` and ``value`` keys. For example:

.. tab-set::

   .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build]
         envs = [
              { name = "VAR_NAME", value = "value" },
              { name = "API_KEY", value = "your_api_key_here" }
         ]

   .. tab-item:: bentofile.yaml

      .. code-block:: yaml

         envs:
           - name: "VAR_NAME"
             value: "value"
           - name: "API_KEY"
             value: "your_api_key_here"

The specified environment variables will be injected into the Bento container.

.. note::

    If you deploy your BentoML Service on BentoCloud, you can either set environment variables through ``envs`` in configuration or the ``--env`` flag when running ``bentoml deploy``. See :ref:`scale-with-bentocloud/deployment/configure-deployments:environment variables` for details.

``conda``
^^^^^^^^^

Conda dependencies can be specified under the ``conda`` field. For example:

.. tab-set::

   .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build.conda]
         channels = ["default"]
         dependencies = ["h2o"]
         pip = ["scikit-learn==1.2.0"]

   .. tab-item:: bentofile.yaml

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

Optionally, you can export all dependencies from a pre-existing conda environment to an ``environment.yml`` file, and provide this file in your ``pyproject.toml`` or ``bentofile.yaml`` file. If it is specified, this file will overwrite any additional option specified.

To export a conda environment:

.. code-block:: bash

    conda env export > environment.yml

To add it in your configuration:

.. tab-set::

   .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build.conda]
         environment_yml = "./environment.yml"

   .. tab-item:: bentofile.yaml

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

.. _docker-configuration:

``docker``
^^^^^^^^^^

BentoML makes it easy to deploy a Bento to a Docker container. It provides a set of options for customizing the Docker image generated from a Bento.

The following ``docker`` field contains some basic Docker configurations:

.. tab-set::

   .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build.docker]
         distro = "debian"
         python_version = "3.11"
         system_packages = ["libblas-dev", "liblapack-dev", "gfortran"]

   .. tab-item:: bentofile.yaml

      .. code-block:: yaml

         docker:
           distro: debian
           python_version: "3.11"
           system_packages:
             - libblas-dev
             - liblapack-dev
             - gfortran

BentoML uses `BuildKit <https://github.com/moby/buildkit>`_, a cache-efficient builder toolkit, to containerize Bentos. BuildKit comes with `Docker 18.09 <https://docs.docker.com/develop/develop-images/build_enhancements/>`_. This means if you are using Docker via Docker Desktop, BuildKit will be available by default. If you are using a standalone version of Docker, you can install BuildKit by following the instructions `here <https://github.com/docker/buildx#installing>`_.

The following sections provide detailed explanations of certain Docker configurations.

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

.. tab-set::

   .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build.python]
         packages = ["nltk"]
         [tool.bentoml.build.docker]
         setup_script = "./setup.sh"

   .. tab-item:: bentofile.yaml

      .. code-block:: yaml

         python:
           packages:
             - "nltk"
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

.. tab-set::

   .. tab-item:: pyproject.toml

      .. code-block:: toml

         [tool.bentoml.build.python]
         packages = ["nltk"]
         [tool.bentoml.build.docker]
         setup_script = "./setup.py"

   .. tab-item:: bentofile.yaml

      .. code-block:: yaml

         python:
           packages:
             - "nltk"
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

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - distro
     - The OS distribution on the Docker image. It defaults to ``debian``.
   * - python_version
     - The Python version on the Docker image. It defaults to the Python version in the build environment.
   * - cuda_version
     - Deprecated. The CUDA version on the Docker image for running models that require GPUs. When using PyTorch or TensorFlow to run models on GPUs, we recommend you directly install them along with their respective CUDA dependencies, using ``pip``. This means you don't need to configure ``cuda_version`` separately. See :doc:`/build-with-bentoml/gpu-inference` for more information.
   * - system_packages
     - The system packages that will be installed in the container.
   * - setup_script
     - A Python or Shell script that will be executed during the Docker build process.
   * - base_image
     - A user-provided Docker base image. This will override all other custom attributes of the image.
   * - dockerfile_template
     - Customize the generated Dockerfile by providing a Jinja2 template that extends the default Dockerfile.

Custom build context
--------------------

For projects that are part of a larger codebase and interact with other local Python
modules or those containing multiple Bentos/Services, it might not be possible to
put all Service definition code and ``pyproject.toml`` (or ``bentofile.yaml``) in the project's root directory.

BentoML allows the placement of the Service definition and ``pyproject.toml`` (or ``bentofile.yaml``) anywhere in the project directory.
In such scenarios, specify the ``build_ctx`` and ``bentofile`` arguments when running the ``bentoml build`` command.

* ``build_ctx``: The build context represents the working directory of your Python project. It will be prepended to the PYTHONPATH during build process,
  ensuring the correct import of local Python modules. By default, it's set to the current directory where the ``bentoml build`` command is executed.
* ``bentofile``: It defaults to the ``pyproject.toml`` (or ``bentofile.yaml``) file in the build context.

To customize their values, use the following:

.. code-block:: bash

    bentoml build -f ./src/my_project_a/bento_fraud_detect.yaml ./src/

Structure
---------

By default, all created Bentos are stored in the BentoML Bento Store, which is essentially a local directory. You can go to a specific Bento directory by running the following command:

.. code-block:: bash

    cd $(bentoml get BENTO_TAG -o path)

Inside the directory, you might see different files and sub-directories depending on the configurations in ``pyproject.toml`` (or ``bentofile.yaml``). A typical Bento contains the following key sub-directories:

* ``src``: Contains files specified in the ``include`` field. These files are relative to user Python code's CWD (current working directory), which makes importing relative modules and file paths inside user code possible.
* ``apis``: Contains API definitions auto-generated from the Service's API specifications.
* ``env``: Contains environment-related files for Bento initialization. These files are generated based on the build options specified in ``pyproject.toml`` (or ``bentofile.yaml``).

.. warning::

   We do not recommend you change files in a Bento directly, unless it's for debugging purposes.
