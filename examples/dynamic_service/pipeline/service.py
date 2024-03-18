from typing import Any
import bentoml


"""The following example is based on the sklearn/pipeline example.

The concept revolves around dynamically constructing service endpoints:

Imagine you have n models ready for production.
When creating your Bento, you may not know in advance which models will be served.
Therefore, you create an endpoint for every available model that can be deployed.

Scenario: You trained hundreds of models.
While they are still in the training pipeline, you want to begin serving your first models already in production.

When constructing Bentos, you require a predefined service.py file. However, the number of endpoints is unknown
during construction of this file. You aim to reuse the same file each time you create a new Bento, without the need
to alter the service definitions repeatedly. Each model should ideally have a route with a unique running index,
for instance. """

def wrap_service_methods(model: bentoml.Model,
                         targets: Any,
                         predict_route: str,
                         predict_name: str,
                         predict_proba_route: str,
                         predict_proba_name: str,
                         ):
    """Wrap models in service methods and annotate as api."""
    @bentoml.api(route=predict_route, name=predict_name)
    async def predict(input_doc: str):
        predictions = await model.predict.async_run([input_doc])
        return {"result": targets[predictions[0]]}

    @bentoml.api(route=predict_proba_route, name=predict_proba_name)
    async def predict_proba(input_doc: str):
        predictions = await model.predict_proba.async_run([input_doc])
        return predictions[0]


    return predict, predict_proba

class_attrs = {} # Empty dict for storing methods
# Manually add api methods to local scope as via locals() method (current scope).
distinct_models = set()
for model in bentoml.models.list():
    distinct_models.add(model.tag.name)
for idx, available_model in enumerate(distinct_models):
    if "twenty_news_group" in available_model:
        bento_model = bentoml.sklearn.get(f"{available_model}:latest")
        target_names = bento_model.custom_objects["target_names"]
        path_predict = f"predict_model_{idx}"
        path_predict_proba = f"predict_proba_model_{idx}"

        class_attrs[path_predict],class_attrs[path_predict_proba] = wrap_service_methods(bento_model,
                                                      target_names,
                                                      predict_route="/"+path_predict,
                                                      predict_name="/"+path_predict,
                                                      predict_proba_route=path_predict_proba,
                                                      predict_proba_name=path_predict_proba,
                                                      )

#  Create class with type and add generated methods
DynamicServiceClass = type(
    "DynamicService", (object,), class_attrs,
)

#  Create Endpoint Service defined in bentofile.yaml
DynamicService = bentoml.service(workers=1, resources={"cpu": "1"})(DynamicServiceClass)