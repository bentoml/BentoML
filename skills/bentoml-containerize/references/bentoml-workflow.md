# How BentoML works — the pipeline behind these skills

Orientation for the whole chain: what each artifact is, where it lives on disk, and
which command turns one into the next. Read this when the project is unfamiliar, when
the user asks what a Bento is, or when something is missing and you need to know which
stage produced it.

```
model or existing code                 ->  a script, a notebook, a FastAPI app, ...
          |  bentoml-create-bento
          v
service.py + runtime environment       ->  the Python project
          |  bentoml build
          v
Bento (~/bentoml/bentos/<name>/<ver>)  ->  standardized, versioned package
          |  bentoml containerize
          v
OCI image (port 3000)                  ->  runs anywhere Docker/OCI runs
          |  push
          v
Registry  ->  Kubernetes | EC2 | any Docker host
```

`bentoml-create-bento` owns the top (project and Bento), `bentoml-containerize` the middle
(build → image → registry), the deploy skills the bottom. Nothing below a stage can be
fixed by editing something above it without rebuilding.

## 1. The Python project

A class decorated with `@bentoml.service` is a **Service**; each `@bentoml.api` method is
an HTTP endpoint at `POST /<method_name>`, with the request and response schemas inferred
from the type annotations. Unannotated parameters become `Any`: no validation, no schema
in `/docs.json`, nothing for a deploy skill's verification request to derive a payload
from.

```python
import bentoml

@bentoml.service
class Summarization:
    def __init__(self) -> None:
        from transformers import pipeline
        self.pipeline = pipeline("summarization")

    @bentoml.api
    def summarize(self, text: str) -> str:
        return self.pipeline(text)[0]["summary_text"]
```

Models are loaded in `__init__`, never per request. A project may hold several Services
wired with `bentoml.depends()`; the one that is built is the **entry service**, it takes
external traffic and calls the others. That graph is what `bentoml-k8s-deploy` turns into
one Deployment + Service per BentoML service.

## 2. The runtime environment

Every dependency the service imports must be declared, in `bentoml.images.Image` (in
code, preferred), `bentofile.yaml`, or `[tool.bentoml.build]` in `pyproject.toml`. The
local virtualenv is not consulted at build time — a package that is only installed
locally is simply absent from the image. Full field reference:
[runtime-environment.md](runtime-environment.md).

Env var **names** are declared here (`envs=[{"name": "HF_TOKEN"}]`); values are injected
at run time and never written into a spec file, a manifest, or an image layer.

## 3. `bentoml build` — the Bento

A **Bento** is the standardized, versioned package of everything needed to serve the
service: source code, the runtime environment spec, model references, and metadata. Each
build produces a new immutable version, so build once and capture the tag.

| Path | Contents |
|---|---|
| `~/bentoml/bentos/<name>/<version>/` | Built Bentos. `bentoml list` shows them (newest first), `bentoml get <tag> -o json` inspects one. |
| `~/bentoml/models/<name>/<version>/` | Models in the local Model Store. `bentoml models list` shows them. |
| `~/bentoml/` | Root of both stores. Set `BENTOML_HOME` to relocate them — needed in CI, where the runner's home directory is ephemeral. |

The default Bento name is the snake_case service class name (`MyService` →
`my_service`). Inside each Bento, `bento.yaml` records the topology: `services[].name`,
`entry_service`, and `services[].dependencies[].service`. Read that file rather than
inferring the topology from `service.py`.

`bentoml build` packages everything under the build context (the directory holding
`service.py`) that matches `include`, minus `.git/`, `.venv/`, `venv/`, `__pycache__/`
and `.DS_Store`. Everything else large — datasets, checkpoints, notebooks — has to be
excluded in `.bentoignore`, or it ends up in the image.

## 4. `bentoml containerize` — the OCI image

The image is a plain OCI image with no BentoML-specific runtime requirements. Its
conventions are fixed, and every downstream skill depends on them:

- HTTP on **port 3000**.
- `/livez` liveness, `/readyz` readiness (models load before ready — allow minutes),
  `/metrics` Prometheus, `/docs.json` OpenAPI, `/` Swagger UI. These paths belong to the
  server; an API method or mounted ASGI route claiming one is shadowed.
- The entrypoint already runs `serve` — never override the command (no `command:` in a
  pod spec; use `args:` if arguments are needed).
- Models are usually baked into the image. Size disks, volumes and memory limits to it.
- `BENTO_PATH` in the image env points at the unpacked Bento (a custom base image can
  move it), which is how `bento.yaml` is read back out of an image.

## 5. Deploy

Once the image is in a registry the target can pull from, deployment is ordinary
container operations: `docker run -p 3000:3000 <image>` on any Docker host,
`bentoml-k8s-deploy` for Kubernetes, `bentoml-ec2-deploy` for plain EC2 instances. The
image tag is the Bento version — pin it, because `latest` makes a rollback guesswork.

Out of scope for these skills: BentoCloud, Yatai, Helm charts, and operators.
