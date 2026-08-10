# SageMaker service adaptation — mechanics, variants, edge cases

Everything in this document was verified against the BentoML source tree and validated
by running the adapted service locally with bentoml 1.4.39 (`bentoml serve` and
`docker run -e BENTOML_PORT=8080 ... serve`), curling `GET /ping` (200) and
`POST /invocations` (correct inference JSON).

## Why each piece of the patch is required

**Port — `BENTOML_PORT=8080`, env-var-only.**
SageMaker runs the container as `docker run IMAGE serve` and offers no way to change the
command — only `ContainerDefinition.Environment`. The BentoML image entrypoint
(`src/bentoml/_internal/container/frontend/dockerfile/entrypoint.sh`) turns the `serve`
argument into `exec bentoml serve "$BENTO_PATH"`, and the `bentoml serve` CLI declares
`--port` with `envvar="BENTOML_PORT"` (`src/bentoml_cli/serve.py`) — so the env var alone
moves the server to 8080. (The entrypoint also maps Heroku-style `PORT` onto
`BENTOML_PORT`; either name works, prefer `BENTOML_PORT`.) Nothing needs to change in the
service code or image for the port.

**`POST /invocations` — `@bentoml.api(route="/invocations")`.**
The v2 SDK `api` decorator accepts `route=` (`src/_bentoml_sdk/decorators.py`); it must
start with `/` (validated in `src/_bentoml_sdk/method.py`). `route=` *replaces* the
method's default route rather than adding an alias — which is why the patch adds a new
delegating method instead of re-routing the user's method: re-routing would silently
remove `POST /<method>` and break every other deploy target.

**Delegation via `.local`.**
`APIMethod` is a descriptor (`src/_bentoml_sdk/method.py`): `self.predict` resolves to a
bound caller carrying a `.local` attribute that always invokes the underlying function
in-process. Use `self.predict.local(...)` in the alias so the call can never be routed
through a service proxy.

**`GET /ping` — Starlette app mounted at `path="/"`.**
`@bentoml.asgi_app(app, path=...)` mounts any ASGI app into the service
(`src/_bentoml_sdk/decorators.py`; `bentoml.mount_asgi_app` is the deprecated alias).
Mounted apps are attached as `PassiveMount` router routes
(`src/_bentoml_impl/server/app.py`, `mount.py`):

- A **bare ASGI callable mounted at `path="/ping"`** behaves like a plain Starlette
  `Mount`: `GET /ping` (no trailing slash) gets a **307 redirect** to `/ping/`.
  Observed locally; SageMaker's health check requires a literal 200, so this FAILS.
- A **Starlette app mounted at `path="/"`** goes through `PassiveMount`, which claims a
  request only when one of the app's own routes matches. `Route("/ping", ...)` matches
  `GET /ping` exactly → 200; every other path falls through to the normal BentoML routes.
  This is the validated form. Mounted routes are appended after the API routes, so
  nothing existing is shadowed either way.

Starlette is a direct BentoML dependency — the patch adds no packages.

## Variant: async primary method

```python
    @bentoml.api(route="/invocations")
    async def invocations(self, text: str) -> dict:
        return await self.predict.local(text)
```

## Variant: wrapper file (zero edits to the user's service.py) — validated

If the user refuses edits to `service.py`, create `service_sagemaker.py` next to it.
Subclass the user's class via `.inner` (the `@bentoml.service` decorator returns a
`Service` object wrapping the original class; inherited `@bentoml.api` methods are
picked up by the new service — verified locally, `/predict` remains served):

```python
# service_sagemaker.py
import bentoml
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from service import MyService          # the user's module and class

async def _sagemaker_ping(request):
    return JSONResponse({})

_sagemaker_ping_app = Starlette(routes=[Route("/ping", _sagemaker_ping, methods=["GET"])])


@bentoml.asgi_app(_sagemaker_ping_app, path="/")
@bentoml.service()                      # copy the user's decorator arguments here
class MyServiceSageMaker(MyService.inner):
    @bentoml.api(route="/invocations")
    def invocations(self, text: str) -> dict:   # mirror the primary method's signature
        return self.predict.local(text)
```

Then build **this** service: point `bentofile.yaml` at
`service: "service_sagemaker:MyServiceSageMaker"` (or pass
`bentoml build service_sagemaker:MyServiceSageMaker` equivalents), and make sure the
`include` list covers both files. Validate it locally the same way as the in-place
patch, naming the wrapper service explicitly:

```bash
BENTOML_PORT=8080 bentoml serve service_sagemaker:MyServiceSageMaker
# then curl GET /ping (expect 200) and POST /invocations (expect real inference JSON)
```

Caveats:

- Decorator arguments (`resources`, `image=`, `envs`, timeouts) are NOT inherited —
  copy them from the user's `@bentoml.service(...)` call.
- The bento name changes (snake_cased subclass name) — carry that through image/repo naming.
- Prefer the in-place patch when acceptable: one source of truth, no duplicated config.

## Multiple API methods

SageMaker real-time endpoints expose exactly one inference route. Ask the user which
method is the SageMaker entrypoint and alias only that one. If they need several, either
deploy one endpoint per method (cost multiplies!) or make `invocations` a dispatcher:

```python
    @bentoml.api(route="/invocations")
    def invocations(self, method: str, payload: dict) -> dict:
        if method == "summarize":
            return self.summarize.local(**payload)
        if method == "classify":
            return self.classify.local(**payload)
        raise ValueError(f"unknown method: {method}")
```

Document the dispatcher's body shape in the final report to the user.

## Input formats

- The JSON body of `POST /invocations` maps keys onto the `invocations` parameters —
  identical to any BentoML v2 API route. `invoke-endpoint --content-type application/json`
  matches this directly.
- **Pydantic-model parameters nest under the parameter name.** This is the most common
  post-deploy invoke failure: with

  ```python
  class Request(pydantic.BaseModel):
      text: str
      max_len: int = 100

  @bentoml.api(route="/invocations")
  def invocations(self, req: Request) -> dict:
      return self.predict.local(req)
  ```

  the body must be `{"req": {"text": "...", "max_len": 50}}` — a flat
  `{"text": "...", "max_len": 50}` returns a 400 validation error (surfaced by
  `invoke-endpoint` as `ModelError` 424). Plain scalar parameters (`text: str`) stay
  top-level keys as shown elsewhere in this skill.
- Binary inputs (images, audio): BentoML v2 file parameters expect multipart or typed
  bodies; the simplest SageMaker-friendly shape is to accept base64 in a JSON string
  field and decode inside `invocations` (SageMaker passes the body through verbatim,
  6 MB max).
- The response is whatever the method returns, serialized by BentoML; the
  `invoke-endpoint` output file receives it verbatim.

## What NOT to do

- Do not add `route="/invocations"` to the user's existing method — it replaces the
  original route (see above).
- Do not try to serve `/ping` with `@bentoml.api` — API routes are POST-only; SageMaker
  pings with GET.
- Do not rely on `/readyz`/`/livez` for SageMaker — SageMaker only calls `/ping`. The
  BentoML health routes keep working and remain useful for local smoke tests.
