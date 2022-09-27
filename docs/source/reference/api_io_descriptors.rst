==================
API IO Descriptors
==================

IO Descriptors are used for describing the input and output spec of a Service API.
Here's a list of built-in IO Descriptors and APIs for extending custom IO Descriptor.

NumPy ``ndarray``
-----------------

.. note::

   The :code:`numpy` package is required to use the :obj:`bentoml.io.NumpyNdarray`.

   Install it with ``pip install numpy`` and add it to your :code:`bentofile.yaml`'s under either Python or Conda packages list.

   Refers to :ref:`Build Options <concepts/bento:Bento Build Options>`.

   .. tab-set::

      .. tab-item:: pip

         .. code-block:: yaml
            :caption: `bentofile.yaml`

            ...
            python:
              packages:
                - numpy

      .. tab-item:: conda

         .. code-block:: yaml
            :caption: `bentofile.yaml`

            ...
            conda:
              channels:
                - conda-forge
              dependencies:
                - numpy


.. autoclass:: bentoml.io.NumpyNdarray
.. automethod:: bentoml.io.NumpyNdarray.from_sample


Tabular Data with Pandas
------------------------

To use the IO descriptor, install bentoml with extra ``io-pandas`` dependency:

.. code-block:: bash

    pip install "bentoml[io-pandas]"

.. note::

   The :code:`pandas` package is required to use the :obj:`bentoml.io.PandasDataFrame`
   or :obj:`bentoml.io.PandasSeries`. 

   Install it with ``pip install pandas`` and add it to your :code:`bentofile.yaml`'s under either Python or Conda packages list.

   Refers to :ref:`Build Options <concepts/bento:Bento Build Options>`.

   .. tab-set::

      .. tab-item:: pip

         .. code-block:: yaml
            :caption: `bentofile.yaml`

            ...
            python:
              packages:
                - pandas

      .. tab-item:: conda

         .. code-block:: yaml
            :caption: `bentofile.yaml`

            ...
            conda:
              channels:
                - conda-forge
              dependencies:
                - pandas

.. autoclass:: bentoml.io.PandasDataFrame
.. automethod:: bentoml.io.PandasDataFrame.from_sample
.. autoclass:: bentoml.io.PandasSeries


Structured Data with JSON
-------------------------
.. note::

   For common structure data, we **recommend** using the :obj:`JSON` descriptor, as it provides
   the most flexibility. Users can also define a schema of the JSON data via a
   `Pydantic <https://pydantic-docs.helpmanual.io/>`_ model, and use it to for data
   validation.

   Make sure to install `Pydantic <https://pydantic-docs.helpmanual.io/>`_ with ``pip install pydantic`` if you want to use ``pydantic``.
   Then proceed to add it to your :code:`bentofile.yaml`'s under either Python or Conda packages list.

   Refers to :ref:`Build Options <concepts/bento:Bento Build Options>`.

   .. tab-set::

      .. tab-item:: pip

         .. code-block:: yaml
            :caption: `bentofile.yaml`

            ...
            python:
              packages:
                - pydantic

      .. tab-item:: conda

         .. code-block:: yaml
            :caption: `bentofile.yaml`

            ...
            conda:
              channels:
                - conda-forge
              dependencies:
                - pydantic

.. autoclass:: bentoml.io.JSON

Texts
-----
:code:`bentoml.io.Text` is commonly used for NLP Applications:

.. autoclass:: bentoml.io.Text

Images
------

To use the IO descriptor, install bentoml with extra ``io-image`` dependency:


.. code-block:: bash

    pip install "bentoml[io-image]"

.. note::

   The :code:`Pillow` package is required to use the :obj:`bentoml.io.Image`.

   Install it with ``pip install Pillow`` and add it to your :code:`bentofile.yaml`'s under either Python or Conda packages list.

   Refers to :ref:`Build Options <concepts/bento:Bento Build Options>`.

   .. tab-set::

      .. tab-item:: pip

         .. code-block:: yaml
            :caption: `bentofile.yaml`

            ...
            python:
              packages:
                - Pillow

      .. tab-item:: conda

         .. code-block:: yaml
            :caption: `bentofile.yaml`

            ...
            conda:
              channels:
                - conda-forge
              dependencies:
                - Pillow

.. autoclass:: bentoml.io.Image

Files
-----
.. autoclass:: bentoml.io.File

Multipart Payloads
------------------

.. note::
    :code:`io.Multipart` makes it possible to compose a multipart payload from multiple
    other IO Descriptor instances. For example, you may create a Multipart input that
    contains a image file and additional metadata in JSON.

.. autoclass:: bentoml.io.Multipart

Custom IODescriptor
-------------------

.. note::
    The IODescriptor base class can be extended to support custom data format for your
    APIs, if the built-in descriptors does not fit your needs.

.. autoclass:: bentoml.io.IODescriptor
