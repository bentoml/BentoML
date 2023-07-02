import bentoml

model_runner = bentoml.models.get("text-classification-pipe").to_runner()

svc = bentoml.Service("text-classification-service", runners=[model_runner])

@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())
async def classify(text: str) -> str:
    results = await model_runner.async_run([text])
    return results[0]
