# Converting existing code into a Bento

The goal is a Bento that produces byte-identical output for a known input, with the original
code still importable. Nothing else counts as a successful conversion.

## Order of operations

1. **Capture the anchor.** Run the existing code on one real input; save input and output to
   disk. If it already has tests, they are the anchor.
2. **Inventory.** Inference entry point · where weights load from · the actual import list ·
   anything stateful (DB, cache, queue) · anything web-only (auth, CORS, static files).
3. **Move the weights first.** `save_model.py` into the model store, or a `HuggingFaceModel`
   reference. Re-run the original against the store path to prove nothing changed.
4. **Wrap the entry point.** One `@bentoml.service` class, one `@bentoml.api` method calling
   the existing function. Import the old module; do not paste its body.
5. **Serve and diff** against step 1. Only now change anything else.
6. **Then** move web extras (§FastAPI), split Services if there is a hardware reason, and pin
   the image.

Resist rewriting the model code during a conversion. Two changed variables make a failed diff
uninterpretable.

## A script or a module

```python
# old: predict.py  (unchanged)
def load(): ...
def predict(model, text): ...
```

```python
# service.py
import bentoml
import predict as impl

@bentoml.service
class Predictor:
    def __init__(self) -> None:
        self.model = impl.load()

    @bentoml.api
    def predict(self, text: str) -> dict:
        return impl.predict(self.model, text)
```

Delete from the copy: `argparse`, `if __name__ == "__main__"`, `print()` progress, `sys.exit`,
any `while True` loop. Keep the file on disk — the Bento can import it.

## A notebook

Split the cells: imports → module top level (heavy ones under `bentoml.importing()`); setup and
model loading → `__init__`; the inference cell → the `@bentoml.api` body; the "try it" cell →
the anchor test. Exploration cells are dropped. `%pip install` lines become
`requirements.txt`.

## A FastAPI app

Two options; pick by whether the app has non-inference routes.

**Inference only** — port the routes and delete the app. The pydantic models carry over
unchanged:

```python
# old: @app.post("/predict") async def predict(req: PredictRequest) -> PredictResponse
@bentoml.api
async def predict(self, req: PredictRequest) -> PredictResponse:      # nested under "req"
async def predict(self, **req: t.Any) -> PredictResponse:             # or flat, with input_spec
```

Note the shape change: a parameter typed with a model nests the payload under the parameter
name. `@bentoml.api(input_spec=PredictRequest)` keeps the original flat request body — use it
when existing clients must keep working.

**Has auth / custom routes / static files / webhooks** — keep the app and mount it under a
prefix:

```python
@bentoml.service
@bentoml.asgi_app(app, path="/v1")        # the app's /healthz becomes /v1/healthz
class Service:
    @bentoml.api
    def predict(self, ...): ...
```

Then: drop `uvicorn.run(...)`; `FastAPI(lifespan=...)`/`@app.on_event("startup")` becomes
`__init__` or `@bentoml.on_startup`; `Depends(get_model)` becomes `self.model`; a route needing
the Service instance uses `Depends(bentoml.get_current_service)`.

Two verified traps:

- **Mount under a prefix**, not `/`. `/`, `/livez`, `/readyz`, `/healthz`, `/metrics` and
  `/docs.json` are the server's; a mounted `/healthz` is shadowed with no warning.
- **Auth expressed as a route dependency stops protecting anything that matters.** The
  inference route is now a `@bentoml.api`, which FastAPI's `Depends` never sees. Re-express it
  as ASGI middleware (`Service.add_asgi_middleware(...)`, § 7 of [recipes.md](recipes.md)) and exclude the health
  paths, or the probes fail.

## A Flask or Django app (WSGI)

WSGI is not ASGI. Port the inference route to `@bentoml.api` — that is nearly always less work
than adapting the app. If the surrounding app must be kept, wrap it in a WSGI-to-ASGI adapter
(e.g. `a2wsgi.WSGIMiddleware`) before `@bentoml.asgi_app`, and verify the adapted routes
yourself; BentoML does not ship or test that adapter.

## An MLflow model

```python
bentoml.mlflow.import_model("my-model", model_uri="runs:/<run_id>/model")   # once
```

```python
@bentoml.service
class Predictor:
    model_ref = bentoml.models.BentoModel("my-model:latest")

    def __init__(self) -> None:
        self.model = bentoml.mlflow.load_model(self.model_ref)   # a pyfunc

    @bentoml.api
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df)
```

The pyfunc flavour keeps MLflow's own signature; annotate the API with the types that signature
actually takes. https://docs.bentoml.com/en/latest/examples/mlflow.html

## A BentoML 1.1 project

1.1's `bentoml.Service(...)` + `Runner` + `@svc.api(input=JSON(), output=JSON())` is replaced by
one class per unit.

| 1.1 | 1.2+ |
|---|---|
| `svc = bentoml.Service("name", runners=[r])` | `@bentoml.service class Name:` |
| `bentoml.models.get(tag).to_runner()` | a second `@bentoml.service`, referenced with `bentoml.depends()` |
| `@svc.api(input=NumpyNdarray(), output=JSON())` | `@bentoml.api def f(self, x: np.ndarray) -> dict:` |
| `runner.predict.run(x)` | `self.dep.predict(x)` |
| IO descriptors (`JSON()`, `Image()`, `Text()`) | type annotations |
| `bentofile.yaml` | `bentoml.images.Image` (the yaml still works) |
| `@svc.on_startup` | `__init__` / `@bentoml.on_startup` |

`bentoml.runner_service(runner)` converts a legacy Runner into a Service with no code changes —
a staging step, not an end state. Old `Runner` docs:
https://docs.bentoml.com/en/v1.1.11/concepts/runner.html

## A containerized custom server

If the model already runs behind its own HTTP server (Triton, a Rust binary, a Node service),
do not port it: run it as the Service's process and let BentoML proxy.

```python
@bentoml.service(cmd=["tritonserver", "--http-port", "8000"], http={"proxy_port": 8000}, workers=1)
class Triton:
    pass
```

The command's port and `http.proxy_port` must match; `$PORT` is not defined for you (§ 9 of
[recipes.md](recipes.md)).

You lose the typed schema and the Swagger UI (BentoML no longer sees the API), so prefer this
only when the server cannot be replaced.

## Verifying parity

```bash
# 1. before: the original code, on the original weights
python -c "import model; print(model.predict_one(model.load_model(), INPUT))" > /tmp/before.json
# 2. after the weights move: the original code, on the model-store copy  <- catches step 3 alone
# 3. after the wrap: the Bento, same request body the old API took
bentoml serve service:Predictor &
curl -s -X POST localhost:3000/predict -H 'Content-Type: application/json' \
     -d @anchor.json > /tmp/after.json
python -c "import json;a=json.load(open('/tmp/before.json'));b=json.load(open('/tmp/after.json'));print('identical' if a==b else (a,b))"
```

Compare parsed JSON, not bytes: BentoML emits `{"score": 1.0}` where FastAPI emits
`{"score":1.0}`, and the diff would be pure noise.

Report the diff to the user rather than adjusting the expected value. Common legitimate
differences: float formatting in JSON, `numpy` scalars rendered as JSON numbers, dict key
order, and a `str` return arriving as `text/plain` instead of a JSON-quoted string. Anything
else is a bug in the conversion.
