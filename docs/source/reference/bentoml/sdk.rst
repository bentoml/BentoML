===========
BentoML SDK
===========

Service decorator
-----------------

.. autofunction:: bentoml.service

.. autofunction:: bentoml.runner_service

.. autofunction:: bentoml.asgi_app

Service API
-----------

.. autofunction:: bentoml.api

Note that when you enable batching, ``batch_dim`` can be a tuple or a single value.

- For a tuple (``input_dim``, ``output_dim``):

  - ``input_dim``: Determines along which dimension the input arrays should be batched (or stacked) together before sending them for processing. For example, if you are working with 2-D arrays and ``input_dim`` is set to 0, BentoML will stack the arrays along the first dimension. This means if you have two 2-D input arrays with dimensions 5x2 and 10x2, specifying an ``input_dim`` of 0 would combine these into a single 15x2 array for processing.
  - ``output_dim``: After the inference is done, the output array needs to be split back into the original batch sizes. The ``output_dim`` indicates along which dimension the output array should be split. In the example above, if the inference process returns a 15x2 array and ``output_dim`` is set to 0, BentoML will split this array back into the original sizes of 5x2 and 10x2, based on the recorded boundaries of the input batch. This ensures that each requester receives the correct portion of the output corresponding to their input.

- If you specify a single value for ``batch_dim``, this value will apply to both ``input_dim`` and ``output_dim``. In other words, the same dimension is used for both batching inputs and splitting outputs.

.. dropdown:: Image illustration of ``batch_dim``

    This image illustrates the concept of ``batch_dim`` in the context of processing 2-D arrays.

    .. image:: ../../_static/img/reference/bentoml/sdk/batch-dim-example.png
       :alt: Batching dimension explanation

    On the left side, there are two 2-D arrays of size 5x2, represented by blue and green boxes. The arrows show two different paths that these arrays can take depending on the ``batch_dim`` configuration:

    - The top path has ``batch_dim=(0,0)``. This means that batching occurs along the first dimension (the number of rows). The two arrays are stacked on top of each other, resulting in a new combined array of size 10x2, which is sent for inference. After inference, the result is split back into two separate 5x2 arrays.
    - The bottom path has ``batch_dim=(1,1)``. This implies that batching occurs along the second dimension (the number of columns). The two arrays are concatenated side by side, forming a larger array of size 5x4, which is processed by the model. After inference, the output array is split back into the original dimensions, resulting in two separate 5x2 arrays.

.. autofunction:: bentoml.task

bentoml.depends
---------------

.. autofunction:: bentoml.depends

bentoml.validators
------------------

.. autoclass:: bentoml.validators.PILImageEncoder
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: bentoml.validators.FileSchema
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: bentoml.validators.TensorSchema
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: bentoml.validators.DataframeSchema
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: bentoml.validators.ContentType
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: bentoml.validators.Shape
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: bentoml.validators.DType
    :members:
    :undoc-members:
    :show-inheritance:
