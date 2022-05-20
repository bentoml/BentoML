==================
API IO Descriptors
==================

IO Descriptors are used for describing the input and output spec of a Service API.
Here's a list of built-in IO Descriptors and APIs for extending custom IO Descriptor.

NumPy ndarray
-------------

.. autoclass:: bentoml.io.NumpyNdarray
.. automethod:: bentoml.io.NumpyNdarray.from_sample


Tabular Data with Pandas
------------------------
.. autoclass:: bentoml.io.PandasDataFrame
.. automethod:: bentoml.io.PandasDataFrame.from_sample
.. autoclass:: bentoml.io.PandasSeries


Structure Data with JSON
------------------------
.. note::
    For common structure data, we recommend using the JSON descriptor, as it provides
    the most flexibility. Users can also define a schema of the JSON data via a
    `Pydantic <https://pydantic-docs.helpmanual.io/>`_ model, and use it to for data
    validation.

.. autoclass:: bentoml.io.JSON

Texts
-----
:code:`bentoml.io.Text` is commonly used for NLP Applications:

.. autoclass:: bentoml.io.Text

Images
------
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