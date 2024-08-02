import bentoml
from typing import Any
from bentoml.io import JSON

river_model_runner = bentoml.mlflow.get("river_arf_model:latest").to_runner()

svc = bentoml.Service("river_model_service", runners=[river_model_runner])

input_spec = JSON.from_sample({'ordinal_date': 736489, 'gallup': 37.843213, 'ipsos': 38.07067899999999, 'morning_consult': 42.318749, 'rasmussen': 40.104692, 'you_gov': 38.636914000000004})

@svc.api(input=input_spec, output=JSON())
async def predict(input_data: dict[str, Any]) -> dict[str, Any]:
    return await river_model_runner.predict.async_run(input_data)
