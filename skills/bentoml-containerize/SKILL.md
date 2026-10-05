---
name: bentoml-containerize
description: >
  Build a local BentoML project into a Bento, containerize it into an OCI/Docker
  image, smoke-test it locally, and push it to a container registry (Docker Hub,
  GHCR, ECR, private registry, kind/minikube local load, or ttl.sh). Use when the
  user asks to "containerize a Bento", "build a Docker image for my BentoML
  service", "package my BentoML service for deployment", "push my Bento image to
  a registry", or as the first step of deploying BentoML to Kubernetes or EC2.
  Does NOT deploy anything itself — hand off to bentoml-k8s-deploy or
  bentoml-ec2-deploy for that.
license: Apache-2.0
compatibility: >-
  Requires the bentoml CLI (>= 1.4), a running Docker daemon, and network access to the target image registry; AWS CLI v2 for ECR pushes.
---

# Containerize a BentoML project and push it to a registry

Build the user's project into a Bento, containerize it, smoke-test the container,
push it. Output: a **pushed image reference** for `bentoml-k8s-deploy` or
`bentoml-ec2-deploy`. Steps in order; never skip the smoke test.

> For production / CI-CD, generate a standalone script bundle (no agent at deploy
> time) with `bentoml-deploy-scriptgen`.

[references/bentoml-workflow.md](references/bentoml-workflow.md) is the map of the whole
pipeline: project -> runtime environment -> Bento -> image -> deploy target, with the
local store paths and the container conventions every later step relies on.

## Step 0 — Preflight checks

```bash
command -v bentoml && bentoml --version                      # 1. BentoML CLI
docker info --format '{{.ServerVersion}} {{.Architecture}}'  # 2. docker daemon (containerize's default backend)
ls service.py bentofile.yaml 2>/dev/null                     # 3. locate the project
find . -maxdepth 3 -name service.py -not -path '*/.venv/*'   #    ...if not in the current dir
```

- Missing CLI: `pip install bentoml` into the user's active environment.
- Docker unreachable: user starts Docker (Docker Desktop / colima); stop until it is.
  Note `Architecture` for Step 3: kernel names, `x86_64` = `amd64`, `aarch64` =
  `arm64`; container platforms use the latter (`linux/amd64`).
- Build context = directory holding `service.py`; run later commands there. If the
  project has neither `bentofile.yaml` nor an `image=` spec, create one in Step 1.

Ask both questions in one round:
1. Which registry? Docker Hub, GHCR, ECR, a private registry, `kind`/`minikube` local
   load (no registry), or `ttl.sh`.
2. Target CPU architecture (cluster nodes / EC2 instance type), `amd64` or `arm64`?
   Most clouds are `linux/amd64`, Apple Silicon defaults to `arm64`, and a mismatch
   gives `exec format error` on the target.

## Step 1 — Verify / complete the runtime environment spec

Read `service.py`; confirm a class decorated with `@bentoml.service`. Its snake_case
name is the default Bento name (`MyService` → `my_service`).

Declare every Python dependency the service imports, in one of two places:

- In code, preferred for new projects:
  `bentoml.images.Image(python_version="3.11").python_packages(...)` passed as
  `@bentoml.service(image=my_image)`.
- Or `bentofile.yaml` beside `service.py`: `service: "service:MyService"`,
  `include: ["*.py"]`, `python: {packages: [...]}`.

Add what the imports need and the spec lacks; `bentoml` is always included. Full
`Image` API (`requirements_file()`, `system_packages()`, `run()`, `base_image`,
`distro`, lockfiles), `bentofile.yaml` fields, env vars and models:
[references/runtime-environment.md](references/runtime-environment.md).

**.bentoignore:** `bentoml build` packages everything under the build context matching
`include` (default `*`), minus always-excluded `.git/`, `.venv/`, `venv/`,
`__pycache__/`, `.DS_Store`. Add gitignore-style patterns (`data/`, `checkpoints/`,
`*.ipynb`) for large dirs.

**Secrets:** never give `envs` a secret value (HF_TOKEN, API keys) in
`bentofile.yaml` or the decorator; values are baked into image layers. Name only,
injected at runtime (Step 4 locally, Kubernetes Secret later).

## Step 2 — Build the Bento

Build **once** from the build context, capturing the tag; `-o tag` prints one
`__tag__:name:version` line. No plain `bentoml build` first: every build makes a new
bento with a fresh version.

```bash
BENTO_TAG=$(bentoml build -o tag | grep '^__tag__:' | sed 's/^__tag__://')
echo "$BENTO_TAG"    # e.g. summarization:6oxk5qvott3lsnry
# bentoml list shows Bentos newest first (-o json for scripting); bentoml get "$BENTO_TAG" -o json inspects one
# Bentos live in ~/bentoml/bentos/<name>/<version>/, models in ~/bentoml/models/ (BENTOML_HOME to relocate)
```

Empty `BENTO_TAG`: rerun plain `bentoml build` (no `-o tag`) for the full error.
Usually a `service:` entry not matching the real module/class name, a package missing
at import time, or a model download failure (gated Hugging Face models need
`export HF_TOKEN=...` in this shell).

## Step 3 — Containerize

Compose `$IMAGE` from the chosen registry's pattern in
[references/registries.md](references/registries.md) BEFORE running this.

```bash
# IMAGE = full registry reference, pushed in Step 5, e.g. docker.io/<user>/summarization:6oxk5qvott3lsnry
bentoml containerize "$BENTO_TAG" -t "$IMAGE"
```

- Without `-t`, the image is tagged as the bento tag. With `-t` (preferred, skips a
  `docker tag`) it is tagged ONLY `$IMAGE`; no `$BENTO_TAG` image is created.
- Docker image names must be lowercase.
- **The image tag must be the bento version.** The deploy skills' `config.yml` holds
  `image:` as a bare URL with NO tag and appends the version, so `$IMAGE` = `<URL for
  config.yml>:<bento version>`. Any other tag deploys an image that was never pushed.
  The registry URL is a user-provided dependency (one their cluster can pull from);
  the tag is not a choice.

**Exception, kind/minikube (no registry):** omit `-t`, nothing is pushed. The image
is named exactly `$BENTO_TAG`, which is what you load into the cluster.

```bash
bentoml containerize "$BENTO_TAG"
IMAGE="$BENTO_TAG"
```

**Cross-architecture:** when the build machine's arch differs from the cluster, pass
the target platform (`--platform=linux/amd64` is an accepted legacy spelling); see
[references/cross-platform.md](references/cross-platform.md).

```bash
bentoml containerize "$BENTO_TAG" -t "$IMAGE" --opt platform=linux/amd64
```

Containerize **bakes every referenced model into the image** (BentoModel,
HuggingFaceModel): large image, long first build for LLM-sized models; gated HF models
again need `HF_TOKEN` exported here.

## Step 4 — Smoke test locally with docker run

The entrypoint serves automatically, no command needed. Skip this step only when the
image platform doesn't match the local machine (cross-arch build), and say so.
Container port is always 3000; map it to an uncommon **host** port (3007 below), since
dev servers often hold 3000.

```bash
docker rm -f bento-smoke 2>/dev/null || true
# Add -e NAME=value per runtime env var (e.g. models downloaded at startup).
docker run -d --name bento-smoke -p 3007:3000 "$IMAGE" || { echo "docker run failed"; exit 1; }

# Confirm it is up BEFORE curling — another process on the host port yields a false READY.
docker ps --filter name=bento-smoke --format '{{.Status}}'   # must show "Up ..."

# Wait for readiness (model loading can take minutes for big models)
for i in $(seq 1 60); do
  if curl -sf http://localhost:3007/readyz > /dev/null; then echo READY; break; fi
  sleep 5
done
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3007/readyz   # expect 200
```

`docker run` fails or `/readyz` never returns 200: read `docker logs bento-smoke`.
Missing dependency → Step 1, missing env var → add `-e`, port conflict → another host
port (container port stays 3000). Clean up afterwards even on failure.

Then exercise one real endpoint. **Mandatory**: a 200 on `/readyz` proves neither that
this is your container nor that the model loaded. Take route and payload from the
`@bentoml.api` methods in `service.py` (`POST /<method_name>`, JSON body of the
parameters) and judge the **response body**, not the status code.

```bash
curl -s -X POST http://localhost:3007/<method_name> \
  -H 'Content-Type: application/json' \
  -d '{"<param>": <value>}'
docker rm -f bento-smoke      # clean up
```

## Step 5 — Push to the user's registry

Logins, naming rules, visibility gotchas (GHCR packages default to private!) and
no-registry paths (`kind load` / `minikube image load`, ttl.sh):
[references/registries.md](references/registries.md). Generic flow once `-t` named the
image:

```bash
docker login <registry-host>       # per-registry specifics in the reference
docker push "$IMAGE"
```

Verify: push output ends with a digest (or `docker manifest inspect "$IMAGE"`).

## Step 6 — Hand off to a deploy skill

- `bentoml-k8s-deploy` — one `deploy/config.yml` rendering one Deployment + Service
  per BentoML service the bento declares.
- `bentoml-ec2-deploy` — the pushed image (typically ECR) on EC2 with docker.

Report and pass to the deploy skill:

| What | Detail |
|---|---|
| Image reference | exact pushed `$IMAGE`, or the name loaded into kind/minikube |
| Image URL and bento version, **separately** | URL without the tag (what `image:` in `config.yml` holds) + the bento version (the tag) |
| Registry access | private? K8s: private registry → `imagePullSecrets`; kind/minikube loaded image → `imagePullPolicy` NOT `Always` (use `IfNotPresent`) |
| Runtime env vars | names only; secret values go into a Kubernetes Secret, not manifests |
| Architecture | the image's target arch |
| Deploy-target facts | HTTP on **port 3000**; **`/livez`** liveness, **`/readyz`** readiness; entrypoint already runs `serve`, so never override the command (K8s: no `command:` in the pod spec) |

**Service topology** is informational; the deploy skill reads `bento.yaml` itself
(local bento store, or from the image under `--skip-build`). Report it anyway: number
of `@bentoml.service` classes, the entry service, the dependency edges, from
`bento.yaml` inside the image (`entry_service`, `services[].name`,
`services[].dependencies[].service`) rather than guessed from `service.py`. Take
`BENTO_PATH` from the image, never hard-coded: a custom base image can move it.

```bash
BENTO_PATH=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$IMAGE" \
    | sed -n 's/^BENTO_PATH=//p' | head -n1)
docker run --rm --entrypoint cat "$IMAGE" "${BENTO_PATH:-/home/bentoml/bento}/bento.yaml"
```

Do **not** suggest Kubernetes object names: derived per BentoML service from the
bento's own service names, an invented name is ignored or wrongly written into the
config. Never derive one from the image repository, where a ttl.sh repo is a random
UUID that may start with a digit, invalid as a DNS-1035 Service name.

## Cleaning up (optional)

When done iterating:

```bash
docker rmi "$IMAGE"              # plus docker rmi "$BENTO_TAG" if built without -t
bentoml delete "$BENTO_TAG" -y
# ECR repositories bill for storage:
aws ecr delete-repository --repository-name <repo-used-in-Step-5> --region "$AWS_REGION" --force
```

ttl.sh images expire on their own; Docker Hub/GHCR images go via their web UIs.
