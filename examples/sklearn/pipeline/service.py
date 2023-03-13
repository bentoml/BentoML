import bentoml
from bentoml.io import JSON
from bentoml.io import Text

bento_model = bentoml.sklearn.get("20_news_group:latest")

target_names = bento_model.custom_objects["target_names"]
model_runner = bento_model.to_runner()

svc = bentoml.Service("doc_classifier", runners=[model_runner])


@svc.api(input=Text(), output=JSON())
async def predict(input_doc: str):
    predictions = await model_runner.predict.async_run([input_doc])
    return {"result": target_names[predictions[0]]}


@svc.api(input=Text(), output=JSON())
async def predict_proba(input_doc: str):
    predictions = await model_runner.predict_proba.async_run([input_doc])
    return predictions[0]
