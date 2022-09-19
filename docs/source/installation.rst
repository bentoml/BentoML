============
Installation
============

Install 🍱 BentoML with your favourite package manager, alongside with whichever deep learning
library you are working with, and you are ready to go!

🍱 BentoML is distributed as a Python Package available `on PyPI <https://pypi.org/project/bentoml/>`_.

* 🍱 BentoML supports Linux/UNIX, Windows, and MacOS.
* 🍱 BentoML is tested with and requires Python 3.7 and above.

Install with pip
----------------

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

Install with pipx
-----------------

If you prefer tools such as `pipx <https://pypa.github.io/pipx/>`_ to install bentoml
into an isolated environment. This has an added benefit that later you can upgrade
bentoml without affecting other projects.

.. code-block:: bash

    pipx install bentoml

    bentoml --help

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

   This requires you to have `Git <https://git-scm.com/>`_, `pip <https://pip.pypa.io/en/stable/installation/>`_, and `Python3.7+ <https://www.python.org/downloads/>`_ installed.

.. seealso::

   For more information on development notes, refer to `our development guide <https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md>`_.

.. warning::

   Make sure that the current directory is not at ``$HOME`` directory, since BentoML will
   create a ``~/bentoml`` directory internally.

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

.. warning::

   You must not remove ``bentoml`` folder after installing in editable mode to keep using
   the library.

After that you can easily update your clone with the latest changes on ``main`` branch
with the following command:

.. code-block:: bash

    cd bentoml
    git pull

Install with conda
------------------

Install 🍱 BentoML with `conda <https://docs.conda.io/en/latest/>`_ via the `conda-forge <https://conda-forge.org/>`_ channel:

.. code-block:: bash

    conda install -c conda-forge bentoml


Deep learning frameworks integration
------------------------------------

BentoML provides first-class support for a list of :doc:`Deep learning frameworks <frameworks/index>`. In order to 
use these integration with BentoML, you will need to install its corresponding package.

For example: :doc:`bentoml.tensorflow <frameworks/tensorflow>` module requires ``tensorflow`` package to be installed.

Additional features
-------------------

To use additional features in BentoML, such as gRPC, S3 support, and more, you will need
to install a variant of BentoML with additional dependencies.

To use gRPC support, use the following command:

.. code-block:: bash

    pip install "bentoml[grpc]"

To use S3 upload support, Image IO support, pydantic validation for JSON, use the
following command:

.. code-block:: bash

    pip install "bentoml[extras]"

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


.. note::

    Historical releases can be found on the `BentoML Releases page <https://github.com/bentoml/BentoML/releases>`_.

.. seealso::

    For the 0.13-LTS releases, see the `0.13-LTS documentation <https://docs.bentoml.org/en/v0.13.1/>`_.

