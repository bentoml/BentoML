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
    cloud (yatai)       BentoCloud Subcommands Groups
    containerize        Containerizes given Bento into an OCI-compliant...
    delete              Delete Bento in local bento store.
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
    serve-grpc          Start a gRPC BentoServer from a given 🍱

Install BentoML from source code
--------------------------------

If you want to install BentoML from the source, run the following command:

.. code-block:: bash

    pip install git+https://github.com/bentoml/BentoML

This command installs the bleeding edge ``main`` version of BentoML, which is useful for staying up-to-date with the latest features and bug fixes. However, the ``main`` version may not always be stable. If you run into any issues, please either create `an issue <https://github.com/bentoml/BentoML/issues/new/choose>`_ or join our community on `Slack <https://l.bentoml.com/join-slack>`_ to get help.

Editable install
----------------

You may want an editable install to:

- Stay up-to-date with the latest features and bug fixes;
- Contribute to the BentoML project and test code changes.

Clone the repository to your local folder and install BentoML with ``pip``:

.. code-block:: bash

    git clone https://github.com/bentoml/bentoml.git
    cd bentoml
    pip install -e .

This command installs BentoML in `editable mode <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_, which allows you to install the project without copying any files. Python links this folder and adds it to Python library paths. This means that any changes to the folder can be tested immediately. For more information, see the `Developer Guide <https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md>`_.

.. warning::

    To keep using the library, you must not remove the ``bentoml`` folder after installing it in editable mode.
