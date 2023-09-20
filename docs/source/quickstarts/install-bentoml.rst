===============
Install BentoML
===============

BentoML is distributed as a Python package available on `PyPI <https://pypi.org/project/bentoml/>`_.
You can install BentoML on Linux/UNIX, Windows, or macOS along with your preferred deep learning library to get started.

This quickstart describes how to install BentoML.

Prerequisites
-------------

- Python 3.8+ and ``pip`` installed. See the `Python downloads page <https://www.python.org/downloads/>`_ to learn more.
- (Optional) `Git <https://git-scm.com/>`_ installed for `editable install <#editable-install>`_.

Install BentoML
---------------

To install BentoML, use the following command:

.. code-block:: bash

   pip install bentoml

To install all additional features in BentoML, such as gRPC and S3 support, use the ``all`` variant. Features can also be installed separately later.

.. code-block:: bash

    pip install "bentoml[all]"

Install BentoML from source code
--------------------------------

If you want to install BentoML from the source, run the following command:

.. code-block:: bash

    pip install git+https://github.com/bentoml/bentoml

This command installs the bleeding edge ``main`` version of BentoML, which is
useful for staying up-to-date with the latest features and bug fixes. However,
the ``main`` version may not always be stable. If you run into any issues, please either
create `an issue <https://github.com/bentoml/BentoML/issues/new/choose>`_ or join our community on
`Slack <https://l.bentoml.com/join-slack>`_ to get help.

Editable install
---------------------

You may want an editable install to:

- Stay up-to-date with the latest features and bug fixes;
- Contribute to the BentoML project and test code changes.

Clone the repository to your local folder and install BentoML with ``pip``:

.. code-block:: bash

    git clone https://github.com/bentoml/bentoml.git
    cd bentoml
    pip install -e .

This command installs BentoML in `editable mode
<https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_,
which allows you to install the project without copying any files. Python links this
folder and adds it to Python library paths. This means that any changes to the
folder can be tested immediately.

.. dropdown:: For users with ``setuptools 64.0.0+``
   :icon: question

   BentoML uses `setuptools <https://setuptools.pypa.io/en/latest/>`_ to build and
   package the project. Since ``setuptools 64.0.0``, setuptools implemented `PEP 660 <https://peps.python.org/pep-0660/>`_, which changes the behavior of editable install in comparison with previous versions.

   Currently, BentoML is not compatible with this new behavior. To install BentoML in editable mode, you have to pass ``--config-settings editable_mode=compat`` to ``pip``.

   .. code-block:: bash

      pip install -e ".[grpc]" --config-settings editable_mode=compat

   See setuptools' `development mode guide <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_ for more information.

.. warning::

   To keep using the library, you must not remove the ``bentoml`` folder after installing it in editable mode.

You can easily update your cloned repository with the latest changes on the ``main`` branch
with the following command:

.. code-block:: bash

    cd bentoml
    git pull

See also
--------

- :doc:`/quickstarts/deploy-a-transformer-model-with-bentoml`
- :doc:`/quickstarts/deploy-a-large-language-model-with-openllm-and-bentoml`
