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

   Refer to :ref:`Build Options <concepts/bento:Bento Build Options>`.

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
.. automethod:: bentoml.io.NumpyNdarray.from_proto
.. automethod:: bentoml.io.NumpyNdarray.from_http_request
.. automethod:: bentoml.io.NumpyNdarray.to_proto
.. automethod:: bentoml.io.NumpyNdarray.to_http_response


Tabular Data with Pandas
------------------------

To use the IO descriptor, install bentoml with extra ``io-pandas`` dependency:

.. code-block:: bash

    pip install "bentoml[io-pandas]"

.. note::

   The :code:`pandas` package is required to use the :obj:`bentoml.io.PandasDataFrame`
   or :obj:`bentoml.io.PandasSeries`. 

   Install it with ``pip install pandas`` and add it to your :code:`bentofile.yaml`'s under either Python or Conda packages list.

   Refer to :ref:`Build Options <concepts/bento:Bento Build Options>`.

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
.. automethod:: bentoml.io.PandasDataFrame.from_proto
.. automethod:: bentoml.io.PandasDataFrame.from_http_request
.. automethod:: bentoml.io.PandasDataFrame.to_proto
.. automethod:: bentoml.io.PandasDataFrame.to_http_response
.. autoclass:: bentoml.io.PandasSeries
.. automethod:: bentoml.io.PandasSeries.from_sample
.. automethod:: bentoml.io.PandasSeries.from_proto
.. automethod:: bentoml.io.PandasSeries.from_http_request
.. automethod:: bentoml.io.PandasSeries.to_proto
.. automethod:: bentoml.io.PandasSeries.to_http_response


Structured Data with JSON
-------------------------
.. note::

   For common structure data, we **recommend** using the :obj:`JSON` descriptor, as it provides
   the most flexibility. Users can also define a schema of the JSON data via a
   `Pydantic <https://pydantic-docs.helpmanual.io/>`_ model, and use it to for data
   validation.

   To use the IO descriptor with pydantic, install bentoml with extra ``io-json`` dependency:

   .. code-block:: bash

      pip install "bentoml[io-json]"

   This will include BentoML with `Pydantic <https://pydantic-docs.helpmanual.io/>`_
   alongside with BentoML

   Then proceed to add it to your :code:`bentofile.yaml`'s under either Python or Conda packages list.

   Refer to :ref:`Build Options <concepts/bento:Bento Build Options>`. We also provide
   :examples:`an example project <pydantic_validation>` using Pydantic for request validation.

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
.. automethod:: bentoml.io.JSON.from_sample
.. automethod:: bentoml.io.JSON.from_proto
.. automethod:: bentoml.io.JSON.from_http_request
.. automethod:: bentoml.io.JSON.to_proto
.. automethod:: bentoml.io.JSON.to_http_response

Texts
-----
:code:`bentoml.io.Text` is commonly used for NLP Applications:

.. autoclass:: bentoml.io.Text
.. automethod:: bentoml.io.Text.from_sample
.. automethod:: bentoml.io.Text.from_proto
.. automethod:: bentoml.io.Text.from_http_request
.. automethod:: bentoml.io.Text.to_proto
.. automethod:: bentoml.io.Text.to_http_response

Images
------

To use the IO descriptor, install bentoml with extra ``io-image`` dependency:


.. code-block:: bash

    pip install "bentoml[io-image]"

.. note::

   The :code:`Pillow` package is required to use the :obj:`bentoml.io.Image`.

   Install it with ``pip install Pillow`` and add it to your :code:`bentofile.yaml`'s under either Python or Conda packages list.

   Refer to :ref:`Build Options <concepts/bento:Bento Build Options>`.

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
.. automethod:: bentoml.io.Image.from_sample
.. automethod:: bentoml.io.Image.from_proto
.. automethod:: bentoml.io.Image.from_http_request
.. automethod:: bentoml.io.Image.to_proto
.. automethod:: bentoml.io.Image.to_http_response

Files
-----

.. autoclass:: bentoml.io.File
.. automethod:: bentoml.io.File.from_sample
.. automethod:: bentoml.io.File.from_proto
.. automethod:: bentoml.io.File.from_http_request
.. automethod:: bentoml.io.File.to_proto
.. automethod:: bentoml.io.File.to_http_response

Multipart Payloads
------------------

.. note::
    :code:`Multipart` makes it possible to compose a multipart payload from multiple
    other IO Descriptor instances. For example, you may create a Multipart input that
    contains a image file and additional metadata in JSON.

.. autoclass:: bentoml.io.Multipart
.. automethod:: bentoml.io.Multipart.from_proto
.. automethod:: bentoml.io.Multipart.from_http_request
.. automethod:: bentoml.io.Multipart.to_proto
.. automethod:: bentoml.io.Multipart.to_http_response

Custom IODescriptor
-------------------

.. note::
    The IODescriptor base class can be extended to support custom data format for your
    APIs, if the built-in descriptors does not fit your needs.

.. autoclass:: bentoml.io.IODescriptor
