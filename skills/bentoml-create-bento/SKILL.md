---
name: bentoml-create-bento
description: >
  Create a BentoML project: the `service.py` whose typed `@bentoml.api` methods become an
  HTTP API, plus its runtime image and a built Bento. Writes one from scratch, or converts
  existing code — a script, a notebook, a FastAPI/Flask app, an MLflow model, a BentoML 1.1
  Runner project. Use for "create a bento", "wrap my model in BentoML", "convert my FastAPI
  app to BentoML", "add an endpoint to my bento", or a bento that will not start, rejects
  valid input, or serves one request at a time.
license: Apache-2.0
compatibility: >-
  Requires Python >= 3.9 and the bentoml CLI >= 1.4 (`pip install bentoml`). Serving needs
  whatever the model itself needs; nothing else. Docker is not needed until bentoml-containerize.
---

# Create a BentoML project

A **Bento** is the artifact `bentoml build` produces; a **project** is the directory it is
built from, normally one `service.py` plus its dependencies. `@bentoml.service` on a class
makes it a **Service** — the unit that gets deployed and scaled, and there may be several in
one bento — and `@bentoml.api` on a method makes it an HTTP endpoint whose **schema is the
method's type annotations**. BentoML supplies the server, the OpenAPI spec, the Swagger UI,
the client, health checks and metrics. You write the class.

Two paths, same steps 2-5: **A** from scratch (§1A) · **B** convert existing code (§1B).

It also covers an existing project: adding an endpoint, switching an API to streaming or
adaptive batching, moving long work to a background task, splitting one Service into several,
and diagnosing a bento that will not start, rejects valid input, or serves one request at a
time (last section).

## What you produce

| File | When | Purpose |
|---|---|---|
| `service.py` | always | The Services and their APIs. The whole surface. |
| `requirements.txt` | always | Runtime deps, referenced by the `Image` and pip-installed locally |
| `.bentoignore` | always | Keeps data, venvs, checkpoints, `__pycache__` out of the bento |
| `test_service.py` | always | The anchor case as a test (§4) |
| `save_model.py` | own weights | One-shot script that puts weights in the model store |
| `bentofile.yaml` / `pyproject.toml` | legacy only | Alternative to `Image`; use `Image` instead ([build options]) |

No Dockerfile, no `uvicorn`/`gunicorn` call, no `main()`, no route table, no request/response
plumbing. If you are writing any of those, you are working against the framework.

## Step 0 — get the anchor case

Before writing anything, get **one concrete input and its expected output** from the user (or
from the existing code's tests/README). This is the anchor: §4 verifies against it. Without it
you can only confirm that a bento returns *something*, which is how a wrong one ships.

If the user cannot supply one, derive it from the model card / existing code and state the
assumption explicitly.

## Step 1A — new project: ask once

Ask these together, with defaults, in one round. Accept "you decide" for any of them.

| Ask | Default when the user has no preference |
|---|---|
| What does it do, and is it one endpoint or several? | One, named after the verb (`summarize`, `classify`) |
| Where do the weights come from: a Hugging Face ID, local files, an existing model-store tag, or no model at all? | HF ID if the user named a model; else no model |
| Input and output types, per endpoint | `str -> str`, or the model's natural pair (§ [io-types]) |
| GPU required? | No |
| Python version and pinned packages? | Your current Python; unpinned packages |
| Response profile: interactive, long-running, token-streaming, or fire-and-forget? | Interactive |

**Do not ask** about workers, threads, replicas, ports, batching, registries or timeouts. The
defaults are correct until something is measured, and every one of them is a one-line change
later.

Then scaffold the file set above and go to §2. Keep the first version to one Service and one
API even if the target is bigger — get it green, then grow.

## Step 1B — convert existing code

Read the existing code first. Find and write down: (1) the inference entry point, (2) where the
weights live, (3) the real dependency list, (4) the anchor case. Then translate:

| In the existing code | In the Bento |
|---|---|
| Module-level model load, or a `load_model()` called per request | `__init__` — runs once per worker |
| A pickle / `.pt` / directory of weights | `save_model.py` once, then a class attribute `BentoModel("name:latest")` |
| `from_pretrained("org/model")` | class attribute `HuggingFaceModel("org/model")`, pass the returned path |
| `@app.post("/predict")` handler | `@bentoml.api def predict(...) -> ...` with annotations |
| Pydantic request/response models | Keep them verbatim — use as a parameter type, or `@bentoml.api(input_spec=Model)` |
| `argparse` / `if __name__ == "__main__"` | Delete. `bentoml serve` is the entry point |
| `uvicorn.run(...)`, gunicorn worker count | Delete; `@bentoml.service(workers=N)` |
| `Dockerfile`, `pip install` lines | `bentoml.images.Image(...)` (§3) |
| Custom routes, static files, webhooks, a UI | Keep the FastAPI app as-is and mount it under a prefix: `@bentoml.asgi_app(app, path="/v1")` |
| Auth / rate limiting as a route dependency | ASGI middleware (`Service.add_asgi_middleware`) — FastAPI `Depends` never sees `@bentoml.api` routes |
| A Flask/Django (WSGI) app | Port the inference route to `@bentoml.api`; a WSGI app needs an ASGI adapter to mount |
| BentoML 1.1 `Runner` / `.to_runner()` | A second `@bentoml.service` + `bentoml.depends()`; `bentoml.runner_service(runner)` is a stopgap |
| Celery / RQ / a background thread | `@bentoml.task` (§ [recipes]) |
| A notebook | Cells to `__init__` (setup) and one `@bentoml.api` (the inference cell) |

Rules for a conversion: **do not rewrite the model code.** Import it, call it, keep it
importable so the old and new paths can be diffed. Convert the smallest slice that serves the
anchor case, verify parity (§4), then move the rest. Detail and per-source playbooks:
[conversion].

## Step 2 — write `service.py`

```python
from __future__ import annotations

import bentoml
from bentoml.models import HuggingFaceModel

with bentoml.importing():           # heavy/optional imports: skipped at build, required at serve
    from transformers import pipeline


@bentoml.service(
    resources={"cpu": "2"},         # deployment hint (see rules), not a local limit
    traffic={"timeout": 30},        # seconds; default 60
)
class Summarization:
    model_path = HuggingFaceModel("sshleifer/distilbart-cnn-12-6")   # class scope, always

    def __init__(self) -> None:     # once per worker
        self.pipeline = pipeline("summarization", model=self.model_path)

    @bentoml.api                    # -> POST /summarize
    def summarize(self, text: str) -> str:
        return self.pipeline(text)[0]["summary_text"]
```

Rules that are not guessable from the API surface:

| Rule | Why it bites |
|---|---|
| Model references (`HuggingFaceModel`, `BentoModel`) go at **class scope**, not in `__init__` | Class scope is what declares the model a dependency of the bento. Inside `__init__` it is not packaged, and deployment fails with model `NotFound` |
| Annotate **every** parameter and the return | The annotations *are* the schema, the OpenAPI spec and the client. An unannotated parameter becomes `Any`: no validation, no docs, no client typing |
| `Optional[X]` requires an explicit default (`= None`); other unions are unsupported | Silent schema surprise otherwise |
| Load models in `__init__`, never inside an API method | Per-request loading is the single most common cause of "BentoML is slow" |
| Service name = class name; `@bentoml.service(name="...")` overrides. API route = `/<method_name>`; `@bentoml.api(route=...)` overrides | Deploy configs, `BENTOML_SERVE_DEPENDS` and clients all key off these names |
| A method name may not start with `__` | Raises at import |
| A **sync** API runs in a thread pool of size `threads` (default **1**) — one request at a time per worker | Concurrency comes from `async def`, or `threads=N`, or `workers=N`. Not from adding replicas |
| Never block inside an `async def` API | It stalls the event loop; `/readyz` starts failing under load. Use `await self.dep.to_async.method(...)`, or `anyio.to_thread.run_sync` |
| `resources={"cpu": ..., "memory": ...}` is a **hint for the deployment platform** — `bentoml serve` does not enforce it. `resources={"gpu": N}` does act locally: it sets `CUDA_VISIBLE_DEVICES` per worker | Expecting a local memory cap leads to debugging the wrong layer. On Kubernetes the real limits come from the deploy config |
| `traffic={"timeout": 60}` is the default | Long generation gets cut off at 60 s with no other explanation |
| Files: take a `pathlib.Path` in; write outputs into `ctx.temp_dir` and return the `Path` | Anything else leaks temp files across requests |
| Raise `bentoml.exceptions.*` subclasses for client-visible errors (`InvalidArgument` -> 400); codes 401, 403 and >= 500 are reserved | Otherwise every failure is a 500 |
| `/`, `/livez`, `/readyz`, `/healthz`, `/metrics`, `/docs.json` are taken by the server | An API or a mounted app route with one of those paths is silently shadowed — mount ASGI apps under a prefix |

Multiple Services in one `service.py`: give each its own `@bentoml.service`, wire them with
`dep = bentoml.depends(Other)` at class scope, and call `self.dep.method(...)` like a local
method. The Service you serve is the **entry** service; BentoML starts its dependencies with
it. Split only for a real reason — different hardware, independent scaling, or a shared
downstream — since each split adds a network hop. Details: [distributed services].

IO types beyond `str`/`int`/`dict` (numpy, pandas, `PIL.Image`, `Path`, pydantic models, root
input, validators, streaming, batching): [io-types]. Config keys: [service-config]. Task
queues, LLM streaming, model composition, FastAPI mounts, Gradio UIs: [recipes].

## Step 3 — declare the runtime environment

```python
image = bentoml.images.Image(python_version="3.11").requirements_file("requirements.txt")

@bentoml.service(image=image)
class Summarization: ...
```

- `Image` is the current API (BentoML >= 1.3.20) and lives next to the code it describes.
  Chainable: `.python_packages(...)`, `.requirements_file(...)`, `.system_packages(...)`,
  `.run("cmd")`, `.run_script("scripts/setup.sh")`, `.build_include(...)`. `.run()` is
  position-sensitive: before `.python_packages()` runs before pip, after it runs after.
- `bentoml` itself is added automatically. Don't list it.
- Versions are locked at build time unless you pass `lock_python_packages=False`.
- A multi-Service bento uses **only the entry Service's image**; per-Service images are not
  supported yet, so put every dependency there.
- `.bentoignore` (gitignore syntax) is what keeps the bento small — always exclude the venv,
  `__pycache__/`, datasets and checkpoints.

## Step 4 — serve and verify (never skip)

```bash
bentoml serve                      # resolves bentofile.yaml -> pyproject.toml -> service.py
bentoml serve service:Summarization --reload -p 3000    # explicit target, dev reload
```

Then, in this order — each check catches a different class of mistake:

```bash
curl -sf localhost:3000/readyz                       # 1. it started and loaded the model
curl -s localhost:3000/docs.json | head -c 400       # 2. the schema is what you intended
curl -s -X POST localhost:3000/summarize \           # 3. the anchor case, judged by its BODY
     -H 'Content-Type: application/json' -d '{"text": "..."}'
```

Judge the **response body** against the anchor from §0, not the status code — a bento that
returns `null`, an empty string or a stack trace with a 200 is a failed verification. For a
conversion, run the original code on the same input and diff the two outputs; report the
comparison, and if they differ, say so rather than adjusting the expectation.

Commit the anchor as a test. No server needed:

```python
import pytest
from starlette.testclient import TestClient
from service import Summarization

@pytest.fixture(scope="session")            # to_asgi() ONCE per process: it registers
def client():                               # Prometheus collectors, and a second call
    with TestClient(Summarization.to_asgi()) as c:   # raises "Duplicated timeseries"
        yield c

def test_summarize(client):                 # TestClient must be a context manager
    r = client.post("/summarize", json={"text": "..."})
    assert r.status_code == 200 and "expected substring" in r.text
```

`Summarization()` can also be instantiated directly and called as plain Python — the fastest
loop while iterating. More: [testing].

## Step 5 — build and hand off

```bash
bentoml build          # -> summarization:xxxxxxxx  (packages the cwd, minus .bentoignore)
bentoml list
bentoml serve summarization:latest      # serve the built artifact, not the source
```

The bento name is the entry Service's name in snake_case (`DiabetesRisk` -> `diabetes_risk`) —
that, not the class name, is what the deploy skills and `bentoml containerize` take.

A built Bento is the input to **`bentoml-containerize`** (image + registry push), then
`bentoml-k8s-deploy` or `bentoml-ec2-deploy`. Tell the user that is the next step; do not
build images or touch a cluster from this skill.

## When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `does not contain a valid bentofile.yaml or service.py` | Wrong cwd, or the file isn't named `service.py` | `cd` to the project, or `bentoml serve mymodule:MyService` |
| Model `NotFound` at deploy, works locally | Model reference created inside `__init__` | Move it to class scope |
| `400` + a pydantic `detail` array on an obviously valid request | Annotation mismatch; a missing `= None` on an `Optional`; a batchable API sent a scalar; a `ContentType`-annotated upload whose client-declared MIME type differs (`curl -F f=@x.csv` sends `application/octet-stream`) | Read the `detail` array, then `/docs.json` — it shows exactly what the server expects |
| Import error at serve time only | A heavy import under `bentoml.importing()` is genuinely missing from the runtime env | Add it to `requirements.txt` / `.python_packages()` |
| Throughput ~1 request at a time | Sync API, `threads=1` | `async def`, or `threads=N`, or `workers=N` |
| Request dies at exactly 60 s | Default `traffic.timeout` | Raise it on the Service |
| `/readyz` flaps under load | Blocking call inside an `async def` | Move to a sync API or `to_async`/`to_thread` |
| Bento build takes forever, artifact is huge | Datasets/venv/checkpoints packaged | `.bentoignore` |
| Dependency Service is never called | Depended-on class not `@bentoml.service`, or `depends()` result not a class attribute | Both are required |
| GPU idle | `workers="cpu_count"` skips GPU assignment | Set an explicit `workers=N` with `resources={"gpu": N}` |

## References

Local, load when the task needs them:

| File | Contents |
|---|---|
| [io-types] | Every supported input/output type, validators, root input, files/images/tensors/dataframes, streaming, batching, what is *not* supported |
| [service-config] | `@bentoml.service` keys: workers, threads, traffic, resources, envs, labels, http, metrics, endpoints, `cmd`; which are local vs. platform-only |
| [recipes] | Copy-pasteable: sklearn/XGBoost, HF transformers, LLM streaming, multi-Service composition, adaptive batching, task queue, FastAPI mount, Gradio UI, external server via `cmd` |
| [conversion] | Per-source playbooks (script, notebook, FastAPI, Flask, MLflow, BentoML 1.1) and the parity procedure |

Upstream docs (authoritative, versioned): [Services], [IO types], [runtime environment],
[model loading], [distributed services], [batching], [tasks], [streaming], [ASGI], [hooks],
[clients], [testing], [build options], [SDK reference], [examples].

[io-types]: references/io-types.md
[service-config]: references/service-config.md
[recipes]: references/recipes.md
[conversion]: references/conversion.md
[Services]: https://docs.bentoml.com/en/latest/build-with-bentoml/services.html
[IO types]: https://docs.bentoml.com/en/latest/build-with-bentoml/iotypes.html
[runtime environment]: https://docs.bentoml.com/en/latest/build-with-bentoml/runtime-environment.html
[model loading]: https://docs.bentoml.com/en/latest/build-with-bentoml/model-loading-and-management.html
[distributed services]: https://docs.bentoml.com/en/latest/build-with-bentoml/distributed-services.html
[batching]: https://docs.bentoml.com/en/latest/get-started/adaptive-batching.html
[tasks]: https://docs.bentoml.com/en/latest/get-started/async-task-queues.html
[streaming]: https://docs.bentoml.com/en/latest/build-with-bentoml/streaming.html
[ASGI]: https://docs.bentoml.com/en/latest/build-with-bentoml/asgi.html
[hooks]: https://docs.bentoml.com/en/latest/build-with-bentoml/lifecycle-hooks.html
[clients]: https://docs.bentoml.com/en/latest/build-with-bentoml/clients.html
[testing]: https://docs.bentoml.com/en/latest/build-with-bentoml/testing.html
[build options]: https://docs.bentoml.com/en/latest/reference/bentoml/bento-build-options.html
[SDK reference]: https://docs.bentoml.com/en/latest/reference/bentoml/sdk.html
[examples]: https://docs.bentoml.com/en/latest/examples/overview.html
