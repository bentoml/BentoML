============
Bento Client
============

BentoML provides a client implementation that can be used to make requests to a BentoML server.

.. note:: This feature is only available if both the client and server are running version 1.0.8 of
   BentoML or newer.

After starting your server, you can initialize a BentoML client by using :obj:`~bentoml.client.Client.from_url`:

.. code-block:: python

    from bentoml.client import Client

    client = Client.from_url("http://localhost:3000")

The client can then be used to make requests to the BentoML server using the api name associated
with the function. For example, for the quickstart service endpoint ``classify``:

.. code-block:: python

    @svc.api(
        input=NumpyNdarray.from_sample(np.array([[4.9, 3.0, 1.4, 0.2]], dtype=np.double)),
        output=NumpyNdarray(),
    )
    async def classify(input_series: np.ndarray) -> np.ndarray:
        return await iris_clf_runner.predict.async_run(input_series)

The client can be used in four ways. Calling the client has been made to be as similar to calling
the API function manually as possible.

.. code-block:: python

    res = client.classify(np.array([[4.9, 3.0, 1.4, 0.2]]))
    res = client.call("classify", np.array([[4.9, 3.0, 1.4, 0.2]]))



Note that even though classify is an ``async`` function, the ``classify`` provided by the client is
still a synchronous function by default. The synchronous ``classify`` and ``call`` *cannot* be
called within a running async event loop.

If inside an async function, use the async versions of the client methods instead:

.. code-block:: python

    res = await client.async_classify(np.array([[4.9, 3.0, 1.4, 0.2]]))
    res = await client.async_call("classify", np.array([[4.9, 3.0, 1.4, 0.2]]))


For multipart requests, all arguments to the function must currently be keyword arguments.

For example, for the service API function:

.. code-block:: python

    @svc.api(input=Multipart(a=Text(), b=Text()), output=JSON())
    def combine(a, b) -> dict[typing.Any, typing.Any]:
        return {a: b}

The client call would look like:

.. code-block:: python

    res = client.combine(a="a", b="b")

.. note:: 

   For all API functions that use ``pydantic_model`` for validation, The output from the client will
   be a dictionary. The keys of the dictionary will be the names of the fields in the output model.
   One can then use ``pydantic_model.parse_obj`` to parse the dictionary into the model.

   .. code-block:: python

      class ModelOutput(pydantic.BaseModel):
          a: str
          b: str

      @svc.api(input=JSON(), output=JSON(pydantic_model=ModelOutput))
      def combine_json(inputs) -> ModelOutput:
          outputs = iris_clf_runner.run(inputs)
          return ModelOutput(**outputs)

   The output of the client would then be:

   .. code-block:: python

      res = client.combine_json({"a": "a", "b": "b"})
      # res = {"a": "a", "b": "b"}
      model_output = ModelOutput.parse_obj(res)

.. note::

   If a custom ``json_encoder`` is used, the output from the client will also be a dictionary. Make sure 
   to use the same ``json_encoder`` if you need to parse the outputs to somewhere else.

   .. code-block:: python

      import json

      res = client.combine_json_with_custom_encoder({"a": "a", "b": "b"})
      # res = {"a": "a", "b": "b"}
      o = json.dumps(res, cls=MyCustomJsonEncoder, ...)
