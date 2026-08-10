---
name: bentoml-containerize
description: >
  Build a local BentoML project into a Bento, containerize it into an OCI/Docker
  image, smoke-test it locally, and push it to a container registry (Docker Hub,
  GHCR, ECR, private registry, kind/minikube local load, or ttl.sh). Use when the
  user asks to "containerize a Bento", "build a Docker image for my BentoML
  service", "package my BentoML service for deployment", "push my Bento image to
  a registry", or as the first step of deploying BentoML to Kubernetes, EC2, or
  SageMaker. Does NOT deploy anything itself — hand off to bentoml-k8s-deploy,
  bentoml-ec2-deploy, or bentoml-sagemaker-deploy for that.
---

# Containerize a BentoML project and push it to a registry

You will take the user's local BentoML project (a `service.py` plus either a
`bentofile.yaml` or an inline `bentoml.images.Image` spec), build a Bento,
containerize it, verify the container actually serves, and push the image to the
registry the user chooses. The output of this skill is a **pushed image
reference** that the deploy skills (`bentoml-k8s-deploy`, `bentoml-ec2-deploy`,
`bentoml-sagemaker-deploy`) consume.

Work through the steps in order. Do not skip the smoke test.

## Step 0 — Preflight checks

Run each check; fix or instruct before continuing.

```bash
# 1. BentoML CLI present?
command -v bentoml && bentoml --version
```
If missing, install it into the user's active environment:
```bash
pip install bentoml
```

```bash
# 2. Docker daemon reachable? (bentoml containerize uses the docker backend by default)
docker info --format '{{.ServerVersion}} {{.Architecture}}'
```
If this fails, tell the user to start Docker (or Docker Desktop / colima) and stop
until it is reachable. Note the reported `Architecture` — you need it in Step 3.
`docker info` prints kernel names: `x86_64` = `amd64`, `aarch64` = `arm64`
(container platforms use the latter, e.g. `linux/amd64`).

```bash
# 3. Locate the project
ls service.py bentofile.yaml 2>/dev/null
# If not in the current directory (e.g. invoked from a repo root):
find . -maxdepth 3 -name service.py -not -path '*/.venv/*'
```
Identify the build context: the directory containing `service.py`. A
`bentofile.yaml` is optional — modern projects define the runtime environment in
code with `bentoml.images.Image` attached via `@bentoml.service(image=...)`.
If neither a `bentofile.yaml` nor an `image=` spec exists, you must create one
(Step 1). All later commands run **from the build context directory**.

Ask the user up front (one round of questions):
1. Which registry should the image go to? Options: Docker Hub, GHCR, ECR, a
   private registry, `kind`/`minikube` local load (no registry), or `ttl.sh`
   (anonymous, ephemeral — good for throwaway tests).
2. What CPU architecture do the target machines run (cluster nodes / EC2
   instance type / SageMaker instance — `amd64` or `arm64`)? Most clouds are
   `linux/amd64`; a laptop building on Apple Silicon defaults to `arm64` —
   mismatch causes `exec format error` on the target.

## Step 1 — Verify / complete the runtime environment spec

Read `service.py`. Confirm there is a class decorated with `@bentoml.service`.
The class name converted to snake_case becomes the default Bento name (e.g.
`class Summarization` → bento `summarization`, `class MyService` → bento
`my_service`).

The runtime environment must declare **every** Python dependency the service
imports. Check one of:

- **In code (preferred for new projects):**
  ```python
  import bentoml

  my_image = bentoml.images.Image(python_version="3.11") \
      .python_packages("torch", "transformers")

  @bentoml.service(image=my_image)
  class MyService: ...
  ```
- **Or `bentofile.yaml`:**
  ```yaml
  service: "service:MyService"
  include:
    - "*.py"
  python:
    packages:
      - torch
      - transformers
  ```

Cross-check the imports in `service.py` against the declared packages and add
anything missing (`bentoml` itself is always included automatically). For the
full `Image` API (`requirements_file()`, `system_packages()`, `run()`,
`base_image`, `distro`, lockfiles), the `bentofile.yaml` field reference, env
vars, and how models are packaged, read
[references/runtime-environment.md](references/runtime-environment.md).

**.bentoignore:** `bentoml build` packages files under the build context
(everything matching `include`, default `*`). `.git/`, `.venv/`, `venv/`,
`__pycache__/`, and `.DS_Store` are always excluded automatically — the file
matters for large data/checkpoint dirs and other irrelevant artifacts. Create a
`.bentoignore` file in the build context with gitignore-style patterns:
```
data/
checkpoints/
*.ipynb
```

**Secrets:** never put secret values (HF_TOKEN, API keys) in `envs` in
`bentofile.yaml` or the service decorator — they get baked into image layers.
Declare the env var name only; the value is injected at runtime (Step 4 locally,
Kubernetes Secret later).

## Step 2 — Build the Bento

From the build context directory, run the build **once**, capturing the tag —
`-o tag` prints a single `__tag__:name:version` line. (Do not run a plain
`bentoml build` first: every build creates a new bento with a fresh
auto-generated version.)

```bash
BENTO_TAG=$(bentoml build -o tag | grep '^__tag__:' | sed 's/^__tag__://')
echo "$BENTO_TAG"    # e.g. summarization:6oxk5qvott3lsnry
```

If `BENTO_TAG` is empty, rerun as plain `bentoml build` (no `-o tag`) purely as
a diagnostic to see the full error output. Common
failures: `service:` entry not matching the actual module/class name, a missing
package at import time, or model download failures (gated Hugging Face models
need `export HF_TOKEN=...` in this shell — see
[references/runtime-environment.md](references/runtime-environment.md)).

You can always list built Bentos with `bentoml list` (newest first;
`bentoml list -o json` for scripting) and inspect one with
`bentoml get "$BENTO_TAG" -o json`.

## Step 3 — Containerize

`bentoml containerize` builds a Docker image tagged **the same as the bento tag**
by default. Use `-t` to name it for the target registry in one shot (preferred —
skips a separate `docker tag`). **Important:** when `-t` is given, the image is
tagged ONLY with `$IMAGE` — no image named `$BENTO_TAG` is created.

Read the chosen registry's section in `references/registries.md` NOW to compose
`$IMAGE` with the right naming pattern before running this command.

```bash
# IMAGE is the full registry reference you'll push in Step 5,
# e.g. docker.io/<user>/summarization:6oxk5qvott3lsnry
bentoml containerize "$BENTO_TAG" -t "$IMAGE"
```

**Exception — kind/minikube (no registry):** there is nothing to push, so omit
`-t`; the image is then named exactly `$BENTO_TAG`, and that is the name you
load into the cluster in Step 5:

```bash
bentoml containerize "$BENTO_TAG"
IMAGE="$BENTO_TAG"
```

Docker image names must be lowercase. Reuse the bento version as the image tag
so images stay traceable to bentos.

**Cross-architecture:** if the build machine's arch differs from the cluster
(e.g. Apple Silicon → amd64 cluster), pass the target platform:

```bash
bentoml containerize "$BENTO_TAG" -t "$IMAGE" --opt platform=linux/amd64
```

(`--platform=linux/amd64` is an accepted legacy spelling of the same option.)
For multi-arch images and buildx details, read
[references/cross-platform.md](references/cross-platform.md).

Note: containerize downloads all referenced models (BentoModel and
HuggingFaceModel) and **bakes them into the image**. Expect a large image and a
long first build for LLM-sized models; gated HF models again need `HF_TOKEN`
exported in this shell.

## Step 4 — Smoke test locally with docker run

The image's entrypoint serves automatically — no command needed. Skip this step
only if the image platform doesn't match the local machine (cross-arch build);
say so explicitly in that case.

Use an uncommon **host** port (3007 below) — host port 3000 is commonly
occupied by dev servers, and probing an occupied port would test the wrong
process. The container port is always 3000.

```bash
docker rm -f bento-smoke 2>/dev/null || true
docker run -d --name bento-smoke -p 3007:3000 "$IMAGE" || { echo "docker run failed"; exit 1; }

# Confirm the container is actually up BEFORE curling — otherwise a
# stale/unrelated process on the host port yields a false READY.
docker ps --filter name=bento-smoke --format '{{.Status}}'   # must show "Up ..."

# Wait for readiness (model loading can take minutes for big models)
for i in $(seq 1 60); do
  if curl -sf http://localhost:3007/readyz > /dev/null; then echo READY; break; fi
  sleep 5
done
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3007/readyz   # expect 200
```

If the service needs runtime env vars (e.g. models downloaded at startup), add
`-e NAME=value` flags to `docker run`.

If `docker run` fails or `/readyz` never returns 200: `docker logs bento-smoke`
and diagnose (missing dependency → back to Step 1; missing env var → add `-e`;
host port conflict → pick another host port, container port stays 3000). Always
run the cleanup command below afterwards — on the failure path too, once you've
captured the logs.

Then exercise one real endpoint — this is **mandatory**, not optional: a 200 on
`/readyz` alone cannot distinguish your container from some other process that
happens to own the port, and it doesn't prove the model actually loaded. Derive
the route and payload from the `@bentoml.api` methods in `service.py`
(endpoints are `POST /<method_name>` with a JSON body of the method's
parameters) and check the **response content** is a plausible result:

```bash
curl -s -X POST http://localhost:3007/<method_name> \
  -H 'Content-Type: application/json' \
  -d '{"<param>": <value>}'
```

Clean up:

```bash
docker rm -f bento-smoke
```

## Step 5 — Push to the user's registry

Log in and push using the registry the user chose in Step 0. Exact login, image
naming rules, visibility gotchas (GHCR packages default to private!), and the
no-registry paths (`kind load` / `minikube image load`, ttl.sh) are in
[references/registries.md](references/registries.md) — read the section for the
chosen registry.

The generic flow (when `-t` in Step 3 already named the image for the registry):

```bash
docker login <registry-host>       # per-registry specifics in the reference
docker push "$IMAGE"
```

Verify the push succeeded (the push output ends with a digest, or pull it back
with `docker manifest inspect "$IMAGE"`).

## Step 6 — Hand off to a deploy skill

This skill's output feeds three sibling deploy skills — pick the one matching
where the user wants to run:

- `bentoml-k8s-deploy` — consumes the pushed image reference and generates
  plain Kubernetes manifests (Deployment + Service).
- `bentoml-ec2-deploy` — consumes the pushed image reference (typically ECR)
  and runs it on an EC2 instance with docker.
- `bentoml-sagemaker-deploy` — consumes an ECR image reference and creates a
  SageMaker endpoint from it.

Report to the user, and pass to the chosen deploy skill:

1. **Image reference**: the exact pushed `$IMAGE` (or the image name loaded into
   kind/minikube).
2. **Registry access**: whether the registry is private. K8s-specific: a
   private registry needs `imagePullSecrets`, and for kind/minikube loaded
   images `imagePullPolicy` must NOT be `Always` (use `IfNotPresent`).
3. **Runtime env vars** the service needs (names only; secret values go into a
   Kubernetes Secret, never into manifests).
4. **Architecture** the image was built for.
5. **Suggested service/deployment name**: the snake_cased bento name with
   underscores converted to hyphens (e.g. `my_service` → `my-service`). Never
   derive it from the image repository — for ttl.sh images the repo is a random
   UUID that may start with a digit, which is an invalid DNS-1035 name for a
   Kubernetes Service.
6. Useful facts for the deploy target: the container serves HTTP on
   **port 3000**; health endpoints are **`/livez`** (liveness) and
   **`/readyz`** (readiness); the image entrypoint already runs `serve`, so no
   command override is needed (K8s-specific: do not set `command:` in the pod
   spec).

## Cleaning up (optional)

When the user is done iterating: remove the local image with
`docker rmi "$IMAGE"` (and `docker rmi "$BENTO_TAG"` if built without `-t`),
delete the bento with `bentoml delete "$BENTO_TAG" -y`. Registry side: ECR
repositories bill for storage — remove with
`aws ecr delete-repository --repository-name <repo-used-in-Step-5> --region "$AWS_REGION" --force`;
ttl.sh images expire on their own; Docker Hub/GHCR images are deleted via their
web UIs.
