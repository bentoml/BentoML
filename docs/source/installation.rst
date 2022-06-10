============
Installation
============

BentoML is distributed as a Python package available `on PyPI <https://pypi.org/project/bentoml/>`_.

* BentoML supports Linux/UNIX, Windows, and MacOS.
* BentoML requires Python 3.7 or above.

.. code-block:: bash

    pip install bentoml --pre

.. note::
    The BentoML version 1.0 is currently under beta preview release, thus :code:`--pre` flag is required.
    For our most recent stable release, see the `0.13-LTS documentation <https://docs.bentoml.org/en/v0.13.1/S>`_.

.. note::
    Historical releases can be found on the `BentoML Releases page <https://github.com/bentoml/BentoML/releases>`_.


You may need to install additional libraries to use certain BentoML modules or features.
For example, the :code:`bentoml.tensorflow` module requires Tensorflow to be installed;
The :code:`bentoml.io.Image` class requires :code:`Pillow` to be installed; and to
export saved models to s3, the :code:`fs-s3fs` package is required.