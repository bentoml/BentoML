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


def wrap_service_methods(
    model: bentoml.Model,
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


@bentoml.service(workers=1, resources={"cpu": "1"})
class DynamicService:
    """Dynamic Service class.

    Note: Variables must not be added in the init function, as the service apis would not be visible in the openapi doc.
    """

    # Manually add api methods to local scope as via locals() method (current scope).
    for idx, available_model in enumerate(bentoml.models.list()):
        if "twenty_news_group" in available_model.tag.name:
            print(f"Creating Endpoint {idx}")
            bento_model = bentoml.sklearn.get(f"{available_model.tag.name}:latest")
            target_names = bento_model.custom_objects["target_names"]
            path_predict = f"predict_model_{idx}"
            path_predict_proba = f"predict_proba_model_{idx}"

            locals()[path_predict], locals()[path_predict_proba] = wrap_service_methods(
                bento_model,
                target_names,
                predict_route=path_predict,
                predict_name=path_predict,
                predict_proba_route=path_predict_proba,
                predict_proba_name=path_predict_proba,
            )

    def __init__(self):
        """Nothing to do here."""
        ...
