===============
Building Bentos
===============

What is a Bento？
----------------

:ref:`Bento 🍱 <reference/core:bentoml.Bento>` is a file archive with all the source
code, models, data files and dependency configurations required for running a
user-defined :ref:`reference/core:bentoml.Service`, packaged into a standardized format.

While :code:`bentoml.Service` standardizes the inference API definition, including the
serving logic, runners initialization and API input, output types.
:code:`Bento` standardizes how to reproduce the required environment for running a
:code:`bentoml.Service` in production.

.. note::
    "Bento Build" is essentially the build process in traditional software development,
    where source code files were converted into standalone artifacts that are ready to
    deploy. BentoML reimagined this process for Machine Learning model delivery, and
    optimized the workflow both for interactive model development and for working with
    automated training pipelines.


The Build Command
-----------------

A Bento can be created with the :ref:`bentoml build <reference/cli:build>` CLI command
with a :code:`bentofile.yaml` build file. Here's an example from the
:doc:`tutorial </tutorial>`:

.. code:: yaml

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

.. code:: bash

    > bentoml build

    INFO [cli] Building BentoML service "iris_classifier:dpijemevl6nlhlg6" from build context "/home/user/gallery/quickstart"
    INFO [cli] Packing model "iris_clf:7drxqvwsu6zq5uqj" from "/home/user/bentoml/models/iris_clf/7drxqvwsu6zq5uqj"
    INFO [cli] Locking PyPI package versions..
    INFO [cli]
         ██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
         ██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
         ██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
         ██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
         ██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
         ╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

    INFO [cli] Successfully built Bento(tag="iris_classifier:dpijemevl6nlhlg6") at "~/bentoml/bentos/iris_classifier/dpijemevl6nlhlg6/"

Similar to :doc:`saving a model </concepts/model>`, a unique version tag will be
automatically generated for the newly created Bento.

It is also possible to customize the Bento version string by specifying it in the
:code:`--version` CLI argument. However this is generally not recommended. Only use it
if your team has a very specific naming convention for deployable artifacts, e.g.:

.. code:: bash

    bentoml build --version 1.0.1

.. note::

    The Bento build process requires importing the :code:`bentoml.Service` object
    defined. This means, the build environment must have all its dependencies installed.
    Support for building from a docker environment is on the roadmap, see :issue:`2495`.

Advanced Project Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^

For projects that are part of a larger codebase and interacts with other local python
modules; Or for projects containing multiple Bentos/Services, it may not be possible to
put all service definition code and :code:`bentofile.yaml` under the project's root
directory.

BentoML allows placing the service definition file and bentofile anywhere in the project
directory. In this case, the user needs to provide the :code:`build_ctx` and
:code:`bentofile` argument to the :code:`bentoml build` CLI command.

build_ctx
    Build context is your Python project's working directory. This is from where you
    start the Python interpreter during development so that your local python modules
    can be imported properly. Default to current directory where the
    :code:`bentoml build` takes place.

bentofile
    :code:`bentofile` is a :code:`.yaml` file that specifies the
    :ref:`concepts/bento:Bento Build Options`. Default to the :code:`bentofile.yaml`
    file under the build context.

They can also be customized via the CLI command, e.g.:

.. code:: bash

    bentoml build -f ./src/my_project_a/bento_fraud_detect.yaml ./src/


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

       .. code:: bash

          > bentoml list

          Tag                               Size        Creation Time        Path
          iris_classifier:nvjtj7wwfgsafuqj  16.99 KiB   2022-05-17 21:36:36  ~/bentoml/bentos/iris_classifier/nvjtj7wwfgsafuqj
          iris_classifier:jxcnbhfv6w6kvuqj  19.68 KiB   2022-04-06 22:02:52  ~/bentoml/bentos/iris_classifier/jxcnbhfv6w6kvuqj

    .. tab-item:: Get

       .. code:: bash

          > bentoml get iris_classifier:latest

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

       .. code:: bash

          > bentoml delete iris_classifier:latest -y

          INFO [cli] Bento(tag="iris_classifier:nvjtj7wwfgsafuqj") deleted


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

        pip install fs-s3fs bentoml
        bentoml import s3://bentoml.com/quickstart/iris_classifier.bento
        bentoml export iris_classifier:latest s3://my_bucket/my_prefix/


Push and Pull
^^^^^^^^^^^^^

`Yatai <https://github.com/bentoml/Yatai>`_ provides a centralized Bento repository
that comes with flexible APIs and Web UI for managing all Bentos created by your team.
It can be configured to store Bento files on cloud blob storage such as AWS S3, MinIO
or GCS, and automatically builds docker images when a new Bento was pushed.

.. code:: bash

  > bentoml push iris_classifier:latest

  Successfully pushed Bento "iris_classifier:nvjtj7wwfgsafuqj"

.. code:: bash

  > bentoml pull iris_classifier:nvjtj7wwfgsafuqj

  Successfully pulled Bento "iris_classifier:nvjtj7wwfgsafuqj"

.. image:: /_static/img/yatai-bento-repos.png
 :alt: Yatai Bento Repo UI


Bento Management API
^^^^^^^^^^^^^^^^^^^^

Similar to :ref:`concepts/model:Managing Models`, equivalent Python APIs are also
provided for managing Bentos:

.. tab-set::

    .. tab-item:: Get

        .. code:: python

            import bentoml
            bento = bentoml.get("iris_classifier:latest")

            print(bento.tag)
            print(bento.path)
            print(bento.info.to_dict())

    .. tab-item:: List

        .. code:: python

            import bentoml
            bentos = bentoml.list()

    .. tab-item:: Import / Export

        .. code:: python

            import bentoml
            bentoml.export_bento('my_bento:latest', '/path/to/folder/my_bento.bento')

        .. code:: bash

            bentoml.import_bento('/path/to/folder/my_bento.bento')

        .. note::

            Bentos can be exported to or import from AWS S3, GCS, FTP, Dropbox, etc. For
            example: :code:`bentoml.export_bento('my_bento:latest', 's3://my_bucket/folder')`

    .. tab-item:: Push / Pull

        If your team has `Yatai <https://github.com/bentoml/Yatai>`_ setup, you can also
        push local Bentos to Yatai, it provides APIs and Web UI for managing all Bentos
        created by your team, stores Bento files on cloud blob storage such as AWS S3, MinIO
        or GCS, and automatically builds docker images when a new Bento was pushed.

        .. code:: bash

            import bentoml
            bentoml.push("iris_classifier:nvjtj7wwfgsafuqj")

        .. code:: bash

            bentoml.pull("iris_classifier:nvjtj7wwfgsafuqj")

    .. tab-item:: Delete

        .. code:: bash

            import bentoml
            bentoml.delete("iris_classifier:nvjtj7wwfgsafuqj")


What's inside a Bento
^^^^^^^^^^^^^^^^^^^^^

It is possible to view the generated files in a specific Bento. Simply use the
:code:`-o/--output` option of the :code:`bentoml get` command to find the file path to
the Bento archive directory.

.. code:: bash

    > cd $(bentoml get iris_classifier:latest -o path)
    > tree
    .
    ├── README.md
    ├── apis
    │   └── openapi.yaml
    ├── bento.yaml
    ├── env
    │   ├── docker
    │   │   ├── Dockerfile
    │   │   ├── entrypoint.sh
    │   │   └── init.sh
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


:bdg-warning:`Warning:` users **should never** change files in the generated Bento
archive, unless it's for debugging purpose.


Bento Build Options
-------------------

Build options are specified in a :code:`.yaml` file, which customizes the final Bento
produced. In this section, we will go over all the build options, including defining
dependencies, configuring files to include, and customize docker image settings.

Service
^^^^^^^

The :code:`service` field is a required which specifies where the
:code:`bentoml.Service` object is defined. In the tutorial example, we have
:code:`service: "service:svc"`, which means:

- :code:`service` refers to the python module (the :code:`service.py` file)
- :code:`svc` refers to the object created in :code:`service.py`, with :code:`svc = bentoml.Service(...)`

This is exact same as how the :ref:`bentoml serve <reference/cli:serve>` command
specifies a :code:`bentoml.Service` target.


Description
^^^^^^^^^^^

The :code:`description` filed allow user to provide custom documentation for a Bento.
The description contents must be plain text, optionally in the
`Markdown <https://daringfireball.net/projects/markdown/syntax>`_ format. Description
can be specified either inline in the :code:`bentofile.yaml`, or via a file path to an
existing text file:

.. code:: yaml

    service: "service.py:svc"
    description: |
        ## Description For My Bento

        Use **any markdown syntax** here!

        > BentoML is awesome!
    includes:
        ...

.. code:: yaml

    service: "service.py:svc"
    description: "file: ./README.md"
    includes:
        ...

.. tip::
    When pointing to a description file, it can be either an absolute path or a relative
    path. The file must exist on the given path upon :code:`bentoml build` command run,
    and for relative file path, the current path is set to the :code:`build_ctx`, which
    default to the directory where :code:`bentoml build` was executed from.


Labels
^^^^^^

:code:`Labels` are key value pairs that are attached to an object. In BentoML, both
:code:`Bento` and :code:`Model` can have labels attached to them. Labels are intended to
be used to specify identifying attributes of Bentos/Models that are meaningful and
relevant to users, but do not directly imply semantics to the rest of the system.

Labels can be used to organize models and Bentos in `Yatai <https://github.com/bentoml/Yatai>`_,
which also allow users to add or modify labels at any time.


Files to include
^^^^^^^^^^^^^^^^

In the example above, the :code:`*.py` is including every Python file from the
:code:`build_ctx`. You can also include other wildcard and directory matching.

.. code:: yaml

    ...
    include:
    - "data/"
    - "**/*.py"
    - "config/*.json"
    - "path/to/a/file.csv"


If the include field is not specified, BentoML, by default, will include
all files under the :code:`build_ctx` directory, besides the ones explicitly set to be
excluded, as shown in the section below.


Files to exclude
^^^^^^^^^^^^^^^^

If the user needs to include a lot of files, another approach is to
only specify which files to be ignored.

The :code:`exclude` field specifies the pathspecs (similar to the :code:`.gitignore`
files) of files to be excluded in the final Bento build. The pathspecs are relative to
the :code:`build_ctx` directory.

.. code:: yaml

    ...
    include:
    - "data/"
    - "**/*.py"
    exclude:
    - "tests/"
    - "secrets.key"

Users can also opt to place a :code:`.bentoignore` file in the :code:`build_ctx`
directory. This is what a :code:`.bentoignore` file would look like.

.. code:: bash

   __pycache__/
   *.py[cod]
   *$py.class
   .ipynb_checkpoints/
   training_data/

.. tip::
    :code:`exclude` is always applied after :code:`include`.


Python Packages
^^^^^^^^^^^^^^^

There are two ways to specify packages in the Bentofile. First,
we can list packages like below.

When left without a version,
pip will just use the latest release from PyPI.

.. code:: yaml

   python:
     packages:
        - numpy
        - "matplotlib==3.5.1"

The user needs to put all required python packages for the Bento Service in a
``requirements.txt``. For a project, you can run ``pip freeze > requirements.txt``
to generate a requirements file to load with BentoML.

.. code:: yaml

   python:
     requirements_txt: "requirements.txt"

Additionally, there are more fields that can help manage larger projects.

.. code:: yaml

   python:
     requirements_txt: "requirements.txt"
     lock_packages: False
     index_url: "https://example.org/"
     no_index: False
     trusted_host: "localhost"
     find_links:
        - "https://test.org/"
     extra_index_url:
        - "https://test.org/"
     pip_args: "--quiet"
     wheels:
        - "./libs/my_package.whl"

PyPI Package Locking
""""""""""""""""""""

By default, when the :code:`bentoml.Service` generates package requirements
from the :code:`Bentofile`, the package versions will be locked for easier
reproducibility. BentoML uses pip-tools to lock the packages.

If the ``requirements.txt`` includes locked packages, or a configuration
you need, set the ``lock_packages`` field to False.

Pip Wheels
""""""""""

If you're maintaining a private pip wheel, it can be included
with the ``wheels`` field.

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
| index_url         | Inputs for the `--index-url` pip argument                                          |
+-------------------+------------------------------------------------------------------------------------+
| no_index          | Whether to include the `--no-index` pip argument                                   |
+-------------------+------------------------------------------------------------------------------------+
| trusted_host      | List of trusted hosts used as inputs using the `--trusted-host` pip argument       |
+-------------------+------------------------------------------------------------------------------------+
| find_links        | List of links to find as inputs using the `--find-links` pip argument              |
+-------------------+------------------------------------------------------------------------------------+
| extra_index_url   | List of extra index urls as inputs using the `≈` pip argument                      |
+-------------------+------------------------------------------------------------------------------------+
| pip_args          | Any additional pip arguments that you would like to add when installing a package  |
+-------------------+------------------------------------------------------------------------------------+
| wheels            | List of paths to wheels to include in the bento                                    |
+-------------------+------------------------------------------------------------------------------------+



Conda Options
^^^^^^^^^^^^^

Similarly to PyPi, you can use Conda to handle dependencies.

.. code:: yaml

   conda:
     dependencies:
        - "scikit-learn==1.2.0"
        - numpy
        - nltk
     channels:
        - "conda-forge"

Here, we need the conda-forge repository to install numpy with conda.
The ``channels`` field let's us specify that to the BentoML service.

In a preexisting environment, running ``conda export`` will generate
an ``environment.yml`` file to be included in the ``environment_yml``
field.

.. code:: yaml

   conda:
     environment_yml: "environment.yml"

Conda Options Table
"""""""""""""""""""
+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| Field            | Description                                                                                                                      |
+==================+==================================================================================================================================+
| environment_yml  | Path to a conda environment file to copy into the bento. If specified, this file will overwrite any additional option specified  |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| channels         | Custom conda channels to use. If not specified will use "defaults"                                                               |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| dependencies     | Custom conda dependencies to include in the environment                                                                          |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+
| pip              | The specific "pip" conda dependencies to include                                                                                 |
+------------------+----------------------------------------------------------------------------------------------------------------------------------+


Docker Options
^^^^^^^^^^^^^^

BentoML makes it easy to deploy a Bento to a Docker container.

Here's a basic Docker options configuration.

.. code:: yaml

   docker:
     distro: debian
     gpu: True
     python_version: "3.8.9"
     setup_script: "setup.sh"

For the ``distro`` options, you can choose from 5.

- debian
- amazonlinux2
- alpine
- ubi8
- ubi7

This config can be explored from `BentoML's Docker page <https://hub.docker.com/r/bentoml/bento-server>`_.

The `gpu` field instructs BentoML to select a Docker base
image that contains NVIDIA drivers and cuDNN library.

For further Docker development, you can also use a ``setup_script``
for the container. This script will run during the ``docker build``
process, as Docker containerizes the image.

For example, with NLP projects you can preinstall NLTK data with:

.. code:: shell
   # ``setup.sh``
   python -m nltk.downloader all

Docker Options Table
""""""""""""""""""""

+-----------------+--------------------------------------------------------------------------------------------------------------------+
| Field           | Description                                                                                                        |
+=================+====================================================================================================================+
| distro          | Configure the particular os distribution on the Docker image ["debian", "amazonlinux2", "alpine", "ubi8", "ubi7"]  |
+-----------------+--------------------------------------------------------------------------------------------------------------------+
| python_version  | Specify which python to include on the Docker image ["3.7", "3.8", "3.9"]                                          |
+-----------------+--------------------------------------------------------------------------------------------------------------------+
| gpu             | Determine if your container will have a gpu. This is not compatible with certain distros                           |
+-----------------+--------------------------------------------------------------------------------------------------------------------+
| devel           | If you want to use the latest main branch from the BentoML repo in your bento                                      |
+-----------------+--------------------------------------------------------------------------------------------------------------------+
| setup_script    | Is a python or shell script that executes during docker build time                                                 |
+-----------------+--------------------------------------------------------------------------------------------------------------------+
| base_image      | Is a user-provided custom docker base image. This will override all other custom attributes of the image           |
+-----------------+--------------------------------------------------------------------------------------------------------------------+

