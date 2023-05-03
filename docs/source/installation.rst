============
Installation
============

🍱 BentoML is distributed as a Python Package available `on PyPI <https://pypi.org/project/bentoml/>`_.
Install BentoML alongside with whichever deep learning library you are working with, and you are ready to go!

* BentoML supports Linux/UNIX, Windows, and MacOS.
* BentoML requires Python 3.7 or above.

.. code-block::

   pip install bentoml

To install all additional features in BentoML, such as gRPC, S3 support, and more, use the ``all`` variant. Features can also be installed separate later.

.. code-block:: bash

    pip install "bentoml[all]"

Install from Source
-------------------

If you want to install BentoML from source, run the following command:

.. code-block:: bash

    pip install git+https://github.com/bentoml/bentoml

This will install the bleeding edge ``main`` version of BentoML. The ``main`` version is
useful for stay-up-to-date with the latest features and bug fixes. However, this means
that ``main`` version is not always stable. If you run into any issues, please either
create `an issue <https://github.com/bentoml/BentoML/issues/new/choose>`_ or join our
`community Slack <https://l.bentoml.com/join-slack>`_ to get help.

Editable Install
----------------

You may want an editable install if:

* You want to stay-up-to-date with the latest features and bug fixes
* You want to contribute to 🍱 BentoML and test code changes

.. note::

   Make sure that you have the following requirements:
    - `Git <https://git-scm.com/>`_
    - `pip <https://pip.pypa.io/en/stable/installation/>`_
    - `Python3.7+ <https://www.python.org/downloads/>`_

.. seealso::

   You're always welcome to make contributions to the project and its documentation. Check out the
    `BentoML development guide <https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md>`_
    and `documentation guide <https://github.com/bentoml/BentoML/blob/main/docs/README.md>`_
    to get started.

Clone the repository to your local folder and install 🍱 BentoML with the following command:

.. code-block:: bash

    git clone https://github.com/bentoml/bentoml.git
    cd bentoml
    pip install -e .

This command will install 🍱 BentoML in `editable mode
<https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_,
which allows you to install the project without copying any files. Python will link this
folder and add it to Python library paths. This means that any changes you make to the
folder will and can be tested immediately.

.. dropdown:: For user using ``setuptools>=64.0.0``
   :icon: question

   BentoML uses `setuptools <https://setuptools.pypa.io/en/latest/>`_ to build and
   package the project. Since ``setuptools>=64.0.0``, setuptools implemented `PEP 660 <https://peps.python.org/pep-0660/>`_, which changes the behavior of editable install in comparison with previous version of setuptools.

   Currently, BentoML is not compatible with this new behavior. To install BentoML in editable mode, you have to pass ``--config-settings editable_mode=compat`` to ``pip``.

   .. code-block:: bash

      pip install -e ".[grpc]" --config-settings editable_mode=compat

   See setuptools' `development mode guide <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_ for more information.


.. warning::

   You must not remove ``bentoml`` folder after installing in editable mode to keep using
   the library.

After that you can easily update your clone with the latest changes on ``main`` branch
with the following command:

.. code-block:: bash

    cd bentoml
    git pull
