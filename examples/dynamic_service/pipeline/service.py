from typing import Any

import bentoml
from bentoml import Runner
from bentoml.io import JSON
from bentoml.io import Text

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


def wrap_service_methods(runner: Runner, targets: Any):
    """Pass Runner and target names, as they are needed in both methods.

    Note: Only passed arguments are available in the methods below, scope is not overwritten.
    """

    async def predict(input_doc: str):
        predictions = await runner.predict.async_run([input_doc])
        return {"result": targets[predictions[0]]}

    async def predict_proba(input_doc: str):
        predictions = await runner.predict_proba.async_run([input_doc])
        return predictions[0]

    return predict, predict_proba


available_model_set = set()
# Add all unique variations of twenty_news_group to the service
for available_model in bentoml.models.list():
    if "twenty_news_group" in available_model.tag.name:
        available_model_set.add(available_model.tag.name)

model_runner_list: [Runner] = []
target_names: [] = []

for available_model in available_model_set:
    bento_model = bentoml.sklearn.get(f"{available_model}:latest")
    target_names.append(bento_model.custom_objects["target_names"])
    model_runner_list.append(bento_model.to_runner())

svc = bentoml.Service("doc_classifier", runners=model_runner_list)

for idx, (model_runner, target_name) in enumerate(zip(model_runner_list, target_names)):
    path_predict = f"predict_model_{idx}"
    path_predict_proba = f"predict_proba_model_{idx}"
    fn_pred, fn_pred_proba = wrap_service_methods(runner=model_runner, targets=target_name)

    svc.add_api(
        input=Text(),
        output=JSON(),
        user_defined_callback=fn_pred,
        name=path_predict,
        doc=None,
        route=path_predict,
    )
    svc.add_api(
        input=Text(),
        output=JSON(),
        user_defined_callback=fn_pred_proba,
        name=path_predict_proba,
        doc=None,
        route=path_predict_proba,
    )
