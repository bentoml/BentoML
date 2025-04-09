============================
Configure template arguments
============================

Starting with BentoML v1.4.8, you can define template arguments for :doc:`Services </build-with-bentoml/services>` using the ``use_arguments()`` API. It allows you to pass dynamic and validated parameters at serve, build, and deploy time.

As these arguments are available directly in Python code, you can substitute values anywhere in ``service.py``, including inside conditionals, loops, or any dynamic logic. It's especially useful when reusing Service templates for different models or configurations.

Define a schema with Pydantic
-----------------------------

Using ``pydantic.BaseModel``, you can set default values for arguments and enable data validation. Here is an example:

.. code-block:: python
   :caption: `service.py`

   from pydantic import BaseModel
   import bentoml

   class BentoArgs(BaseModel):
       model_name: str
       gpu: int = 8
       gpu_type: str = "nvidia-h200-141gb"

   args = bentoml.use_arguments(BentoArgs)

You can then reference the arguments just like regular Python variables.

.. code-block:: python
   :caption: `service.py`

   import bentoml

   @bentoml.service(
       resources={
           "gpu": args.gpu,
           "gpu_type": args.gpu_type
       }
   )
   class LLM:
       model = bentoml.models.HuggingFaceModel(args.model_name)
       ...

Define a schema without Pydantic
--------------------------------

You can use ``use_arguments()`` directly without a Pydantic model. This returns a ``types.SimpleNamespace`` object with all argument values, but without validation or default value support.

.. code-block:: python
   :caption: `service.py`

   import bentoml

   args = bentoml.use_arguments()

   @bentoml.service(resources={"gpu": int(args.gpu)})
   class LLM:
       model = bentoml.models.HuggingFaceModel(args.model_name)
       ...

Provide argument values
-----------------------

After setting the template arguments, you can supply their values dynamically through the CLI when running commands like ``bentoml serve``, ``bentoml build``, ``bentoml deploy`` (when passing the Bento path), and ``bentoml code``. For example:

.. code-block:: bash

   bentoml build --arg model_name=meta-llama/Llama-3.3-70B-Instruct --arg gpu=4
   bentoml serve --arg model_name=deepseek-ai/DeepSeek-V3

.. warning::

   If a required argument is missing, BentoML will raise an error when running the command.
