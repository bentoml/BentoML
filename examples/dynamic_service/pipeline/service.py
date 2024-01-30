import bentoml
from bentoml.io import JSON
from bentoml.io import Text

"""The following example is based on the example sklearn/pipeline.

The idea of dynamically building the service endpoints:

Imaging you have n models ready for production. When building your bento, you do not actually know, which models should
be served, so you create a endpoint for every model that is available for deployment.

Scenario: You are training hundreds of models, while still are in the training pipeline, you already want to serve your
first models in production.

When building bentos, you need a predefined service.py file - but with an unknown number of endpoints when building.
You want to reuse a single file everytime when creating a new bento, without changing the service definitions each time.
Every model should have (for example) a route with a running index.
"""


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


bento_model = bentoml.sklearn.get("twenty_news_group:latest")

target_names = bento_model.custom_objects["target_names"]

# Imaging we have different models, which need the same predict and predict_proba implementations
model_runner_list: [Runner] = [bento_model.to_runner(), bento_model.to_runner()]

svc = bentoml.Service("doc_classifier", runners=[model_runner])

for idx, model_runner in enumerate(model_runner_list):
    path_predict = f"predict_model_{idx}"
    path_predict_proba = f"predict_proba_model_{idx}"
    fn_pred, fn_pred_proba = setMethod(runner=model_runner, targets=target_names)

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
