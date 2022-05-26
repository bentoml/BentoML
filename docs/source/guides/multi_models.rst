=======================
Serving Multiple Models
=======================

Serving multiple models in the same workflow is a pretty straightforward pattern in
BentoML’s prediction framework. Simply instantiate multiple runners up front and pass
them to the service that’s being created. Each runner/model will automatically run with
it’s own resources as configured. If no configuration is passed, then BentoML will
choose the optimal amount of resources to allocate for each runner.

.. code:: python

    import asyncio
    import bentoml
    import PIL.Image

    import bentoml
    from bentoml.io import Image, Text

    transformers_runner = bentoml.transformers.get("sentiment_model:latest").to_runner()
    ocr_runner = bentoml.easyocr.get("ocr_model:latest").to_runner()

    svc = bentoml.Service("sentiment_analysis", runners=[transformers_runner, ocr_runner])

    @svc.api(input=Image())
    async def classify(input: PIL.Image.Image) -> str:
        ocr_text = await ocr_runner.run(input)
        return await transformers_runner.run(ocr_text)


It’s as simple as creating 2 runners and using them together in your prediction
endpoint. An async endpoint is preferred in many cases so that the primary event loop is
yielded when waiting on IO.

Multiple dependent models
-------------------------

In cases where certain steps may be dependent on one another, the :code:`asyncio.gather`
method can be used to await the completion of multiple model results. For example if you
were running 2 models side by side to compare the results, you could await both as
follows:

.. code-block:: python

    import asyncio
    import PIL.Image

    import bentoml
    from bentoml.io import Image, Text

    preprocess_runner = bentoml.Runner(MyPreprocessRunnable)
    model_a_runner = bentoml.xgboost.get('model_a:latest').to_runner()
    model_b_runner = bentoml.pytorch.get('model_b:latest').to_runner()

    svc = bentoml.Service('inference_graph_demo', runners=[
        preprocess_runner,
        model_a_runner,
        model_b_runner
    ])

    @svc.api(input=Image(), output=Text())
    async def predict(input_image: PIL.Image.Image) -> str:
        model_input = await preprocess_runner.async_run(input_image)

        results = asyncio.gather(
            model_a_runner.async_run(model_input),
            model_b_runner.async_run(model_input),
        )

        return post_process(result)


Once each model completes, the results can be compared and logged as a post processing
step.