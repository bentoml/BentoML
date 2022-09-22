============
Installation
============

Install 🍱 BentoML with your favourite package manager, alongside with whichever deep learning
library you are working with, and you are ready to go!

🍱 BentoML is distributed as a Python Package available `on PyPI <https://pypi.org/project/bentoml/>`_.

* BentoML supports Linux/UNIX, Windows, and MacOS.
* BentoML requires Python 3.7 or above.

.. code-block::

   pip install --user bentoml


Install using virtual environment
---------------------------------

You should install BentoML in a `virtual environment <https://docs.python.org/3/library/venv.html>`_. If you are not familar with virtual environment, refers to this
`guide <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`_. Virtual environment are a great way to manage dependencies and
avoid compatibility issues among different projects.

Start create a virtual environment in your project directory

.. code-block:: bash

    python3 -m venv ./venv

.. tip::

   Alternatively, you can also use `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ that provides more features than the built-in venv module.

Activate the virtual environment:

.. tab-set::

    .. tab-item:: MacOS/UNIX

        .. code-block:: bash

            source ./venv/bin/activate

    .. tab-item:: Windows

        .. code-block:: powershell

            .\venv\Scripts\activate

Now, you can install BentoML via pip:

.. code-block:: bash

    pip install bentoml

Additional features
-------------------

To use additional features in BentoML, such as gRPC, S3 support, and more, you will need
to install a variant of BentoML with additional dependencies.

For gRPC support, use the following command:

.. code-block:: bash

    pip install "bentoml[grpc]"

For all AWS-related features, use the following command:

.. code-block:: bash

    pip install "bentoml[aws]"

For all :ref:`Image IO <reference/api_io_descriptors:Images>`, use the following command:

.. code-block:: bash

    pip install "bentoml[io-image]"

For all :ref:`Pandas IO <reference/api_io_descriptors:Tabular Data with Pandas>`, use the following command:

.. code-block:: bash

    pip install "bentoml[io-pandas]"

To use external tracing exporter such as `Jaeger <https://www.jaegertracing.io/>`_, `Zipkin <https://zipkin.io/>`_, `OpenTelemetry Protocol <https://opentelemetry.io/docs/reference/specification/protocol/exporter/>`_,
use the following command:

.. tab-set::

    .. tab-item:: Jaeger

        .. code-block:: bash

            pip install "bentoml[tracing-jaeger]"

    .. tab-item:: Zipkin

        .. code-block:: bash

            pip install "bentoml[tracing-zipkin]"

    .. tab-item:: OpenTelemetry Protocol

        .. code-block:: bash

            pip install "bentoml[tracing-otlp]"

To use all the above features, use the following command:

.. code-block:: bash

    pip install "bentoml[all]"

.. tip::

   The additional dependencies syntax can also be applied to all of the above installation methods

   .. code-block:: bash

      # editable install
      pip install -e ".[grpc,tracing-jaeger]"

Install from source
-------------------

If you want to install BentoML from source, run the following command:

.. code-block:: bash

    pip install git+https://github.com/bentoml/bentoml

This will install the bleeding edge ``main`` version of BentoML. The ``main`` version is
useful for stay-up-to-date with the latest features and bug fixes. However, this means
that ``main`` version is not always stable. If you run into any issues, please either
create `an issue <https://github.com/bentoml/BentoML/issues/new/choose>`_ or join our
`community Slack <https://l.linklyhq.com/l/ktOX>`_ to get help.

Editable install
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

Install using conda
-------------------

Install 🍱 BentoML with `conda <https://docs.conda.io/en/latest/>`_ via the `conda-forge <https://conda-forge.org/>`_ channel:

.. code-block:: bash

    conda install -c conda-forge bentoml


.. note::

    Historical releases can be found on the `BentoML Releases page <https://github.com/bentoml/BentoML/releases>`_.

.. seealso::

    For the 0.13-LTS releases, see the `0.13-LTS documentation <https://docs.bentoml.org/en/v0.13.1/>`_.

