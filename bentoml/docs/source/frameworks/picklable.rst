===============
Picklable Model
===============

For custom ML models created with pure Python code, a simple way to make it work with
the BentoML workflow is via ``bentoml.picklable_model``.

Below is an example of saving a Python function as a model:

.. code:: python

    from typing import List
    import numpy as np
    import bentoml

    def my_python_model(input_list: List[int]) -> List[int]:
        return np.square(np.array(input_list))

    # `save_model` saves a given python object or function
    saved_model = bentoml.picklable_model.save_model(
        'my_python_model',
        my_python_model,
        signatures={"__call__": {"batchable": True}}
    )
    print(f"Model saved: {saved_model}")


Load the model back to memory for testing:

.. code:: python

    loaded_model = bentoml.picklable_model.load_model("my_python_model:latest")

    loaded_model([1, 2, 3])
    # out: array([1, 4, 9])


Load the model as a local Runner to test out its inference API:

.. code:: python

    runner = bentoml.picklable_model.get("my_python_model:latest").to_runner()
    runner.init_local()
    runner.run([7])


Full code example can be found at `Gallery: Custom Python Model <https://github.com/bentoml/BentoML/tree/main/examples/custom_python_model>`_.
