============
Installation
============

BentoML is distributed as a Python package available on `PyPI <https://pypi.org/project/bentoml/>`_. You can install BentoML on Linux/UNIX, Windows, or macOS along with your preferred deep learning library to get started.

This document describes how to install BentoML.

Prerequisites
-------------

- Python 3.8+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- (Optional) We recommend you create a virtual environment for dependency isolation. See the `Conda documentation <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or the `Python documentation <https://docs.python.org/3/library/venv.html>`_ for details.
- (Optional) `Git <https://git-scm.com/>`_ installed for `editable install <https://docs.bentoml.com/en/latest/quickstarts/install-bentoml.html#editable-install>`_.

Install BentoML
---------------

To install BentoML, use the following command:

.. code-block:: bash

    pip install bentoml

To verify your installation, run the following to see :

.. code-block:: bash

    bentoml -h

Expected output:

.. code-block:: bash

    Usage: bentoml [OPTIONS] COMMAND [ARGS]...

    ██████╗ ███████╗███╗   ██╗████████╗ ██████╗ ███╗   ███╗██╗
    ██╔══██╗██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗████╗ ████║██║
    ██████╔╝█████╗  ██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║██║
    ██╔══██╗██╔══╝  ██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║██║
    ██████╔╝███████╗██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║███████╗
    ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚══════╝

    Options:
    -v, --version  Show the version and exit.
    -h, --help     Show this message and exit.

    Commands:
    build               Build a new Bento from current directory.
    cloud               BentoCloud Subcommands Groups.
    containerize        Containerizes given Bento into an OCI-compliant...
    delete              Delete Bento in local bento store.
    deploy              Create a deployment on BentoCloud.
    deployment          Deployment Subcommands Groups
    env                 Print environment info and exit
    export              Export a Bento to an external file archive
    get                 Print Bento details by providing the bento_tag.
    import              Import a previously exported Bento archive file
    list                List Bentos in local store
    models              Model Subcommands Groups
    pull                Pull Bento from a remote Bento store server.
    push                Push Bento to a remote Bento store server.
    serve (serve-http)  Start a HTTP BentoServer from a given 🍱

Install BentoML from source code
--------------------------------

If you want to install BentoML from the source, run the following command:

.. code-block:: bash

    pip install git+https://github.com/bentoml/BentoML

This command installs the bleeding edge ``main`` version of BentoML, which is useful for staying up-to-date with the latest features and bug fixes. However, the ``main`` version may not always be stable. If you run into any issues, please either create `an issue <https://github.com/bentoml/BentoML/issues/new/choose>`_ or join our community on `Slack <https://l.bentoml.com/join-slack>`_ to get help.

Editable install
----------------

Install BentoML in editable mode is useful when you are testing your code changes in a local copy of the BentoML project. First, Clone the repository locally and install BentoML with ``pip install -e``:

.. code-block:: bash

    git clone https://github.com/bentoml/bentoml.git
    cd bentoml
    pip install -e .

This command installs BentoML in `editable mode <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_, which allows you to test any local code changes immediately by importing the ``bentoml`` library again or running a BentoML CLI command. For more information, see the `Developer Guide <https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md>`_.

.. note::

    In editable mode, Python links this folder and adds it to Python library paths. To keep using the library, you must not move or delete the ``bentoml`` folder after installing it in editable mode.

Deploy With Custom Branch
-------------------------

In order to build and deploy a Bento that's using a custom fork of BentoML, add the repo URL to your packages list defined in ``bentofile.yaml``. For example, to use BentoML from the main branch:

.. code-block:: yaml

    python:
        packages:
        - bentoml @ git+https://github.com/bentoml/BentoML.git@main

During ``bentoml build``, the specificed branch will be downloaded from repo url, built and packaged as a wheel file in the Bento created.
