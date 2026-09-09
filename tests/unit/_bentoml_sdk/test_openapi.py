from __future__ import annotations

import bentoml


def test_task_cancel_openapi_operation_name() -> None:
    @bentoml.service
    class TaskService:
        @bentoml.task
        def predict(self, value: int) -> int:
            return value

    spec = TaskService.openapi_spec.asdict()
    retry_operation = spec["paths"]["/predict/retry"]["post"]
    cancel_operation = spec["paths"]["/predict/cancel"]["put"]

    assert retry_operation["x-bentoml-name"] == "predict_retry"
    assert retry_operation["operationId"] == "TaskService__predict_retry"
    assert cancel_operation["x-bentoml-name"] == "predict_cancel"
    assert cancel_operation["operationId"] == "TaskService__predict_cancel"
