import bentoml
from bentoml.io import Text, JSON

bento_model = bentoml.sklearn.get("20_news_group:latest")

target_names = bento_model.custom_objects["target_names"]
model_runner = bento_model.to_runner()

svc = bentoml.Service('doc_classifier', runners=[ model_runner ])

@svc.api(input=Text(), output=JSON())
def predict(input_doc: str):
    prediction = model_runner.predict.run([input_doc])[0]
    return {"result": target_names[prediction]}

@svc.api(input=Text(), output=JSON())
def predict_proba(input_doc: str):
    predictions = model_runner.predict_proba.run([input_doc])[0]
    return predictions
