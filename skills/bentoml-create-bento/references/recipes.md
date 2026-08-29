# Recipes

Working shapes to copy. Each is complete except for the model logic. Official example projects
for every domain (LLM, diffusion, embeddings, CV, audio, RAG, agents):
https://docs.bentoml.com/en/latest/examples/overview.html

## 1. Your own weights, via the model store

Save once, reference by tag. Keeps weights out of git and out of the build context, and lets
the deployment fetch them at image-build time instead of at startup.

```python
# save_model.py — run once: python save_model.py
import bentoml, joblib
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier().fit(X, y)
with bentoml.models.create(name="iris-clf") as ref:      # -> iris-clf:<version>
    joblib.dump(model, ref.path_of("model.pkl"))
```

```python
# service.py
import bentoml, numpy as np
from bentoml.models import BentoModel

@bentoml.service(image=bentoml.images.Image(python_version="3.11")
                 .requirements_file("requirements.txt"))
class IrisClassifier:
    model_ref = BentoModel("iris-clf:latest")            # class scope

    def __init__(self) -> None:
        import joblib
        self.model = joblib.load(self.model_ref.path_of("model.pkl"))

    @bentoml.api
    def classify(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)
```

`bentoml models list` / `get` / `delete` manage the store. Framework helpers
(`bentoml.sklearn.save_model`, `bentoml.xgboost.load_model`, …) exist for many libraries but are
not required — a directory of files and `joblib` work fine.

## 2. Hugging Face model

```python
import bentoml
from bentoml.models import HuggingFaceModel

with bentoml.importing():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

@bentoml.service(envs=[{"name": "HF_TOKEN"}])            # gated models only
class Sentiment:
    model_path = HuggingFaceModel("distilbert-base-uncased-finetuned-sst-2-english")

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

    @bentoml.api
    def classify(self, text: str) -> dict[str, t.Any]:      # label + score
        ...
```

`HuggingFaceModel` returns a **local path string**; pass it wherever you would pass a repo ID.
Downloads happen at build time, so container start-up does not wait on the Hub.

## 3. Streaming LLM output

```python
@bentoml.service(traffic={"timeout": 600}, workers=1, resources={"gpu": 1})
class LLM:
    model_path = HuggingFaceModel("meta-llama/Llama-3.1-8B-Instruct")

    def __init__(self) -> None:
        self.engine = build_engine(self.model_path)       # vLLM, TensorRT-LLM, transformers…

    @bentoml.api
    async def generate(self, prompt: str, max_tokens: int = 512) -> AsyncGenerator[str, None]:
        async for chunk in self.engine.stream(prompt, max_tokens=max_tokens):
            yield chunk
```

`workers=1` per GPU, a long `traffic.timeout`, and `async` throughout. For a real, maintained
vLLM setup (OpenAI-compatible routes, continuous batching, quantization) start from
https://github.com/bentoml/BentoVLLM instead of writing the engine glue by hand.

## 4. Multiple Services

```python
@bentoml.service(resources={"cpu": "1"})
class Embedder:
    @bentoml.api
    def embed(self, texts: list[str]) -> np.ndarray: ...

@bentoml.service(resources={"gpu": 1})
class Reranker:
    @bentoml.api
    def rank(self, query: str, docs: list[str]) -> list[float]: ...

@bentoml.service()                                        # the entry Service
class Search:
    embedder = bentoml.depends(Embedder)                  # class scope
    reranker = bentoml.depends(Reranker)

    @bentoml.api
    async def search(self, query: str) -> list[str]:
        vec, scores = await asyncio.gather(               # run both hops in parallel
            self.embedder.to_async.embed([query]),
            self.reranker.to_async.rank(query, self.candidates(query)),
        )
        ...
```

- Serve the entry Service (`bentoml serve service:Search`); BentoML starts the others.
- `self.dep.method(...)` is a sync call; `self.dep.to_async.method(...)` is awaitable — use it
  in `async def` APIs, and to fan out with `asyncio.gather`.
- Depend on something already running elsewhere: `bentoml.depends(url="http://host:3000")`, or
  `bentoml.depends(deployment="name")` on BentoCloud. Pass `on=TheClass` too for typing.
- Inter-Service payloads are pickled. Expose only the entry Service publicly.

## 5. Adaptive batching with a per-request façade

```python
class Item(BaseModel):
    image: Path
    threshold: float

@bentoml.service
class Detector:
    @bentoml.api(batchable=True, max_batch_size=16, max_latency_ms=500)
    def detect(self, items: list[Item]) -> list[dict]: ...

@bentoml.service
class API:                                                # entry: normal one-at-a-time shape
    detector = bentoml.depends(Detector)

    @bentoml.api
    async def detect(self, image: Path, threshold: float = 0.5) -> dict:
        out = await self.detector.to_async.detect([Item(image=image, threshold=threshold)])
        return out[0]
```

## 6. Fire-and-forget work

```python
@bentoml.service
class Renderer:
    @bentoml.task
    def render(self, prompt: str) -> Path: ...
```

Generated routes, relative to the API's route: `POST /render/submit`, `GET /render/status`,
`GET /render/get`, `POST /render/retry`, `PUT /render/cancel` (note the methods: retry is POST,
cancel is PUT).

```python
task = client.render.submit(prompt="...")
task.get_status().value    # pending | in_progress | completed | failed | canceled
task.get()                 # the result
task.retry()               # a new task with the same input
```

Identical body to an `@bentoml.api` — only the decorator changes. Results are kept ~24 h. A
task cannot return a generator. Locally the queue is in-process and `cancel` is unsupported
("task cancellation is not supported in local development server"), so exercise cancellation
against a real deployment.

## 7. Keep an existing FastAPI app

The conversion path for a service that already has custom routes, webhooks, a UI or static
files. The FastAPI app is mounted whole; the `@bentoml.api` methods stay first-class endpoints.
Auth is the exception — see the second bullet below.

```python
from fastapi import Depends, FastAPI
app = FastAPI()

@bentoml.service
@bentoml.asgi_app(app, path="/v1")                # everything in `app` moves under /v1
class Service:
    @bentoml.api                                  # still POST /predict
    def predict(self, text: str) -> str: ...

    @app.get("/custom")                           # inside the class: `self` works
    def custom(self):
        return {"model": self.name}

@app.get("/other")                                # outside: inject the instance
async def other(svc: Service = Depends(bentoml.get_current_service)):
    ...
```

Two things that are easy to get wrong here:

- **Mount under a prefix.** `/`, `/livez`, `/readyz`, `/healthz`, `/metrics` and `/docs.json`
  belong to the server and win — a mounted app's `/healthz` is silently shadowed and returns
  the built-in response.
- **The mounted app's route dependencies do not protect `@bentoml.api` routes.** FastAPI
  `Depends(...)` only guards FastAPI's own routes. Cross-cutting concerns (auth, rate limits)
  belong in ASGI middleware, which covers everything:

```python
class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path in ("/livez", "/readyz", "/healthz", "/metrics"):
            return await call_next(request)          # never gate the probes
        if request.headers.get("x-api-key") != os.environ["API_KEY"]:
            return JSONResponse({"detail": "bad key"}, status_code=401)
        return await call_next(request)

Service.add_asgi_middleware(ApiKeyMiddleware)        # after the class definition
```

## 8. Gradio UI on the same server

```python
import bentoml, gradio as gr

def ui_fn(text: str) -> str:
    return bentoml.get_current_service().summarize(text)

io = gr.Interface(fn=ui_fn, inputs="text", outputs="text")

@bentoml.service
@bentoml.gradio.mount_gradio_app(io, path="/ui")
class Summarization: ...
```

Needs `gradio` in the image. Serves the demo at `/ui` next to the API.

## 9. An external server binary

When the real server is not Python, or is a pre-built binary: BentoML runs it and proxies to it.

```python
@bentoml.service(cmd=["my-server", "--port", "9000"], http={"proxy_port": 9000}, workers=1)
class External:
    pass
```

- **The command must listen on `http.proxy_port` (default 8000)** — that is where BentoML
  proxies. Hard-code the two to the same number.
- `$VAR` in `cmd` is expanded from the **process** environment, and BentoML defines neither
  `PORT` nor `BENTOML_HOST`: `cmd=[..., "--port", "$PORT"]` dies at startup with
  `KeyError: 'PORT'` unless you supply it yourself (e.g. via `envs=[{"name": "PORT", ...}]`).
- `workers` defaults to `min(16, cpu/2)` for a custom command and only worker 1 starts the
  process; the rest wait on a health check. Set `workers=1` unless the binary can share a port.
- Compute the command at run time with `def __command__(self) -> list[str]`; rewrite the
  metrics it exposes with `def __metrics__(self, original: str) -> str`.

## 10. Calling the service

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000", server_ready_timeout=30) as client:
    print(client.summarize(text="..."))          # method name == API name

async with bentoml.AsyncHTTPClient("http://localhost:3000") as client:
    async for chunk in client.generate(prompt="..."):    # streaming APIs yield
        print(chunk, end="")
```

Root-input APIs take a positional argument. `curl` works equally well; the Swagger UI at `/`
lists every endpoint with its schema.
