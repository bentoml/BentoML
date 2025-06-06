import bentoml

test_model = bentoml.models.get("testmodel")
# get by alias
another_model = bentoml.models.get("another")
svc = bentoml.legacy.Service(
    "test-bento-service-with-models", models=[test_model, another_model]
)
