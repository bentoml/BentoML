---
name: bentoml-k8s-deploy
description: Deploy a containerized BentoML service to a vanilla Kubernetes cluster using plain kubectl manifests (no Helm, no operators, no BentoCloud/Yatai). Takes a pushed container image (built by the bentoml-containerize skill or `bentoml containerize`), discovers the bento's service topology, writes one annotated `config.yml` for the deployment, renders one Deployment + Service per BentoML service (plus optional HPA/Ingress) from it, applies them in dependency order, and verifies the rollout with a real inference request. Use when the user says things like "deploy my BentoML service to Kubernetes", "deploy this bento image to my cluster", "run my bento on k8s", "create k8s manifests for my bento", "split my bento services into separate pods", or "expose my BentoML service in Kubernetes".
---

# Deploy a BentoML service to vanilla Kubernetes

You will take a **pushed container image reference** (e.g. `ghcr.io/acme/summarization:v1`,
produced by the `bentoml-containerize` skill), discover which BentoML services the bento
contains, write a single annotated **`config.yml`** into the project, **render** plain
Kubernetes manifests from it, apply them with `kubectl`, and verify the deployment end to
end.

**`deploy/config.yml` is the deployment config — the one file the user edits.** It holds
the image, the cluster/namespace, and one block per BentoML service (replicas, resources,
probes, env, autoscaling, exposure). The manifests under `k8s/` are **rendered output**:
derived, regenerable, not hand-owned. "Change the deployment" means *edit `config.yml`,
re-render, re-apply* — never *edit the YAML under `k8s/`*.

**One Kubernetes Deployment per BentoML service.** A bento can package several
`@bentoml.service` classes wired together with `bentoml.depends()`; each becomes its own
Deployment + Service so it can be sized, scaled and scheduled independently. A
**single-service bento is the degenerate case** and works exactly the same way: one
service block, one Deployment, one Service, `--service-name` still passed, no dependency
wiring. Do not special-case it.

Derived from `config.yml`, never asked of the user and never hand-maintained:

| Derived | From |
|---|---|
| Rollout order | topological sort of `services.<Name>.depends` — deepest tier first, alphabetical within a tier |
| `BENTOML_SERVE_DEPENDS` | `depends` + each dependency's slug + `kubernetes.namespace` |
| Labels and selectors | each service's slug |
| ClusterIP on every non-entry Service | `entry: false` (an RCE invariant, see below) |
| Omitting `spec.replicas` | `autoscaling.enabled: true` (an HPA and a replica count fight on every apply) |

> **Rendering and validation are not implemented here.** They live in the deploy bundle
> shipped by the sibling `bentoml-deploy-scriptgen` skill, which this skill copies into the
> project (Step 4) and drives with `--render-only` / `--check-only`. One renderer, one
> validator, one `config.yml` — the interactive path and the CI/CD path are the same
> machinery, which is why what you review here is what CI applies later. Running deploys
> from CI needs nothing further: `deploy/` is already committed and runs without an agent.

Facts that are always true for images built by `bentoml containerize`:
- The HTTP server listens on **port 3000** in every pod, whichever service it serves.
- Health endpoints: **`/livez`** (liveness), **`/readyz`** (readiness). `/metrics` serves Prometheus metrics.
- The entrypoint accepts `start-http-server` and appends `$BENTO_PATH` itself. Always use
  **`args:`**, **never `command:`** — `command:` bypasses the entrypoint's venv activation
  and the container dies immediately.
- `args: ["start-http-server", "--service-name", "<ServiceName>"]` makes the pod serve
  exactly one BentoML service and spawn **no sibling processes**.
- **Never set `BENTOML_RUNNER_MAP` / `BENTOML_SERVE_RUNNER_MAP`** on a pod. The serving
  process overwrites it, and any dependency it cannot resolve is then instantiated
  **in-process** — every pod silently loads every model and still reports healthy. Wire
  dependencies with `BENTOML_SERVE_DEPENDS` only (which is what `depends` renders into).
- `@bentoml.service(resources={"cpu": ..., "memory": ...})` is **inert in open-source
  BentoML** — it constrains nothing at serve time. That is precisely why the values in
  `config.yml` are load-bearing: they are the only thing that constrains the pod.
- Models are usually baked into the image; models downloaded at runtime (e.g. gated HF
  models) need env vars such as `HF_TOKEN` via a Kubernetes Secret.

Safety rules (apply throughout):
- **Never apply anything before the user has confirmed the kubectl context AND namespace.**
- Never `kubectl delete` anything this skill did not create in this session.
- Never write secret values into `config.yml` or any file — secrets are created only with
  `kubectl create secret ... --from-literal`, and referenced by name via
  `env_from_secrets` / `kubernetes.image_pull_secret`.
- Never give a **non-entry** service a Service type other than `ClusterIP`: inter-service
  traffic is `application/vnd.bentoml+pickle`, i.e. unauthenticated pickle
  deserialization. Exposing it outside the cluster is remote code execution. `expose:` and
  `ingress:` therefore belong only under the entry service, and the renderer rejects them
  elsewhere.

## Step 0 — Preflight checks

Run, and stop with a clear message if any check fails:

```bash
kubectl version --client        # kubectl installed?
python3 -c 'import yaml; print("pyyaml", yaml.__version__)'   # config.yml needs PyYAML
kubectl config get-contexts     # list contexts; note the current one (*)
```

`config.yml` is YAML and Python's stdlib has no YAML parser, so **PyYAML is required**
(`pip install pyyaml`; `bentoml` already depends on it, so any machine that can build a
bento has it).

Then **ask the user which context to use — never assume the current context is the
intended cluster**, even if there is only one. Show them the list and get an explicit
answer. Use `--context <ctx>` on every subsequent kubectl command (do not silently
switch their current context with `kubectl config use-context`), and record it as
`kubernetes.context`.

With the confirmed context, verify connectivity and permissions:

```bash
kubectl --context <ctx> get nodes -L kubernetes.io/arch   # cluster reachable? note node arch
kubectl --context <ctx> auth can-i create deployment -n <ns>   # run once the namespace is known (Step 3)
```

If the image was built on Apple Silicon and the `ARCH` column above shows `amd64`
nodes, warn the user now: the image must have been containerized with
`--opt platform=linux/amd64` (`image.platform` in the config), otherwise pods will crash
with "exec format error".

## Step 1 — Discover the service topology

Everything downstream (how many service blocks, which one is the entry, what `depends`
says, what defaults to pre-fill) comes from the bento's own metadata. Do this before
asking the user anything about sizing.

**Primary source — `bento.yaml` inside the built image.** It carries the whole topology
without importing user code:

```bash
docker run --rm --entrypoint cat <image> /home/bentoml/bento/bento.yaml
```

`/home/bentoml/bento` is the default `BENTO_PATH`. If a custom base image moved it, read
it from the image config first and substitute:

```bash
docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' <image> | grep '^BENTO_PATH='
```

If the image is only in a remote registry, `docker pull <image>` first (or run the same
`cat` through whatever runtime is available — `podman run --rm --entrypoint cat ...`,
or `nerdctl`).

Read these fields:

| Field | Use |
|---|---|
| `name` | bento name → `project.name`, `app.kubernetes.io/part-of` label, secret name prefix |
| `entry_service` | the BentoML service name that receives external traffic → the one block with `entry: true`; the only one that may be exposed and the only one verify targets |
| `services[].name` | one `services:` block per entry in this list; the block **key is this name verbatim** |
| `services[].dependencies[].service` | the DAG edges → that service's `depends:` list (the value is the **BentoML service name** of the callee) |
| `services[].config` | that service's declared `resources` / `workers` / `traffic` → pre-filled defaults (Step 2) |

Note that `services[]` is **not** in dependency order (the entry service is often first).
You do not need to record an order anywhere — it is derived from `depends` — but report it
so the user can sanity-check the DAG.

**Fallback — the image is not built yet.** `bento.yaml` is **authoritative**; reading the
source is best-effort, so build the image and re-check before shipping anything you
derived this way. Read the user's `service.py` (or whatever module the bento's `service:`
field names) and extract:
- every class decorated with `@bentoml.service(...)` → the service list, with the
  decorator kwargs as the declared config;
- **the service name is the decorator's `name=` kwarg when present, and only the class
  name otherwise** — BentoML resolves it as `name or inner.__name__`. So
  `@bentoml.service(name="sentiment-v2") class Sentiment` is the service
  `sentiment-v2`, not `Sentiment`. Getting this wrong breaks two things with very
  different failure modes: `--service-name` fails **loudly** (the server cannot find the
  service), but the dependency key fails **silently** — an unmatched key falls back to
  instantiating the dependency in-process. Always grep for `name=` in every
  `@bentoml.service(...)` before deriving names;
- every `x = bentoml.depends(Other)` class attribute → a `depends` entry for `Other`, whose
  service name is resolved the same way (follow `Other` to its own decorator's `name=`);
- `@bentoml.service(path_prefix="/x")` if present → that service's `/livez`, `/readyz`
  **and** `/metrics` all move under `/x`, so the rendered probe paths (and any HPA
  scrape path) must be prefixed to match. **The renderer has no config knob for this** —
  flag it to the user and treat that service as a `manifests_dir` case, or drop the
  prefix; do not leave the probes at the root;
- the class named by the build target (`bentoml build service:TextPipeline`, or the
  `service:` key in `pyproject.toml`/`bentofile.yaml`) → the entry service. If that is
  ambiguous, the entry service is the one nothing else depends on; if several qualify,
  ask the user.

Then say plainly which topology you found, e.g.:

```
bento text_pipeline: 2 services
  TextPipeline (entry) -> depends on Sentiment
  Sentiment
Rollout order (derived): Sentiment, then TextPipeline
```

For a wider or deeper DAG, report one line per service with its DIRECT dependencies and
mark the tiers, e.g.:

```
bento gateway: 4 services (entry_service: Gateway)
  Gateway   (entry) -> Enricher, Sentiment      [fan-out]
  Enricher          -> Tokenizer                [middle tier — carries its OWN depends]
  Sentiment         -> (leaf)
  Tokenizer         -> (leaf, 2 hops from the entry)
Rollout order (derived): Sentiment, Tokenizer, Enricher, Gateway
```

Several topological orders are valid; the derived one is **deepest tier first, then
alphabetically within a tier** (hence `Sentiment` before `Tokenizer` above). Report that
order, and let the renderer produce it — do not hand-maintain a list anywhere.

### Slug (object name) rule — DNS-1035

Each service gets a `<slug>`: the BentoML service name snake_cased, then `_` → `-`.
`TextPipeline` → `text-pipeline`, `Sentiment` → `sentiment`, `TTSService` → `tts-service`.
That is the renderer's default for `services.<Name>.slug`; write it explicitly in
`config.yml` so the user can see and change it.

```bash
python3 -c '
import re,sys
def slug(n):
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", n)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower().replace("_", "-")
print(slug(sys.argv[1]))' TextPipeline
```

Kubernetes `Service` objects require an RFC 1035 label — **stricter than most K8s
names**: lowercase alphanumerics and `-`, **must start with a letter**, must end with an
alphanumeric, max 63 chars. The slug names the Service, so:

- Every slug must match `^[a-z]([-a-z0-9]*[a-z0-9])?$`; the renderer validates this and
  exits 2 naming `services.<Name>.slug`, but check it while writing the config.
- Slugs must be **unique** across the bento's services. If two collide, ask the user.
- Never derive the slug from the image repo basename (a ttl.sh UUID like `1f0d3c2a-…`
  starts with a digit and means nothing). Derive it from the BentoML service name.
- **`kubectl apply --dry-run=client` does NOT reliably catch DNS-1035 violations for
  Services** — a bad name passes dry-run, then the real apply rejects the Service while
  the Deployment succeeds, leaving a half-applied state (see Step 6). Never rely on
  dry-run for this.

## Step 2 — One round of questions (bento-wide, then per service)

Ask everything in **one round**: the bento-wide parameters, then a **per-service table
with defaults already pre-filled from each service's `config` in `bento.yaml`**. Let the
user accept everything in one go or edit individual cells. Then write `config.yml` (Step
4) — from that point on, further changes are config edits, not new questions.

Bento-wide:

| Parameter | Config key | Default | Notes |
|---|---|---|---|
| Image reference | `image.registry` + `image.repository` (+ tag at deploy time) | (required) | Full pushed ref incl. registry + tag. If the user only ran `bentoml containerize`, the image may exist only locally — it must be pushed, or loaded with `kind load` / `minikube image load` and declared as `registry_type: none` + `local_image_preloaded: true` (see `references/private-registries.md`). |
| Namespace | `kubernetes.namespace` | `default` | Offer to create a dedicated one. All services of the bento go in the **same** namespace — dependency DNS names are rendered with it. |
| Secrets | `services.<Name>.env_from_secrets` | none | Tokens/keys (e.g. `HF_TOKEN`). Created in Step 3; only the Secret **name** ever reaches the config. |
| Private registry? | `kubernetes.image_pull_secret` | detect/ask | If pulling needs auth → imagePullSecret, Step 3. See `references/private-registries.md`. |
| Exposure (entry service only) | `services.<Entry>.expose` / `.ingress` | port-forward (`ClusterIP`) | One of: port-forward, NodePort, LoadBalancer, Ingress. Read `references/exposure-options.md` first; check what the cluster supports (`kubectl --context <ctx> get ingressclass`, cloud LB availability). Non-entry services are always ClusterIP — do not offer a choice. |
| Verification call | `verify.inference` | derived | The entry service's route + a real payload + a substring that must appear in the answer (Step 7). |

Per service (one row per `services[]` entry):

| Parameter | Config key | Default | Notes |
|---|---|---|---|
| Replicas | `replicas` | `1` | Scale each service independently; the bottleneck is rarely the entry service. |
| CPU request / limit | `resources.requests.cpu` / `resources.limits.cpu` | `config.resources.cpu` if declared, else `"500m"` / `"2"` | If the decorator declares `resources={"cpu": "1"}`, pre-fill request `"1"` and limit `"1"` (or `"2"` for burst) and **say why**: that declaration does nothing in OSS BentoML, so the config is the only place it takes effect. **Always quoted strings** — see the quoting rule below. |
| Memory request / limit | `resources.requests.memory` / `resources.limits.memory` | `config.resources.memory` if declared, else `"1Gi"` / `"4Gi"` | Size to the model: OOMKilled pods usually mean the limit is below the model's real footprint. |
| GPU | `resources.limits["nvidia.com/gpu"]` | `config.resources.gpu` if declared, else none | `resources.gpu` is the one decorator resource key that does something in OSS (it sets `CUDA_VISIBLE_DEVICES`), so it must be matched here or the pod gets no device. Read `references/gpu-scheduling.md` and confirm the cluster advertises the resource; pair with `node_selector`/`tolerations`. |
| Workers | `config_overrides.workers` | `config.workers` if declared, else `1` | Server-side knob, rendered into `BENTOML_CONFIG_OVERRIDES`. Interacts with the CPU limit: N workers want ~N cores. |
| Timeout | `config_overrides.traffic.timeout` | `config.traffic.timeout` if declared, else `60` | Also server-side. Raising a **dependency's** timeout is the one case a per-service block cannot express — see `references/customization.md`. |
| Autoscaling | `autoscaling.*` | `enabled: false` | Opt-in, per service; ask for min/max and the target (CPU % or in-flight requests per pod). The renderer then omits `spec.replicas` for that service. See `references/customization.md`. |
| Env vars (plain) | `env` | none | Non-sensitive config only. |
| Node placement | `node_selector`, `tolerations` | none | GPU pools, arch pinning, tainted node groups. |

**Probe-timeout floor (do not lower it).** `probes.readiness_timeout_seconds` defaults to
`6`, and **6 is a floor, not a default to tune down**. When a service's `/readyz` fans out
to a `bentoml.depends()` dependency, BentoML gives that dependency a **hard-coded
5-second** budget, so `/readyz` can legitimately take just over 5 s. At `3` the kubelet
hangs up first: a dependency that answers in 4 s leaves the calling pod **permanently
unready**, at startup and forever after, with nothing in the pod's own logs to explain it.
The 5 s is a framework constant, not a tunable — `runner_probe.timeout` is inert on
BentoML 1.2+. Raise the value for a deep dependency chain; never lower it, and keep leaf
services on the same value so that adding a dependency later does not silently break them.
The liveness probe stays at `timeoutSeconds: 3` (not configurable) because `/livez` is
pod-local and never cascades.

**Quantity-quoting rule (do not skip this).** Every resource quantity — `cpu`, `memory`,
`nvidia.com/gpu` — is a **quoted string** in `config.yml` and in the rendered manifest:
`cpu: "1"`, `cpu: "500m"`, `memory: "4Gi"`, `nvidia.com/gpu: "1"`. An unquoted whole
number (`cpu: 1`) is a YAML integer; the API server stores the quantity as the string
`"1"`, so the strategic-merge patch sees a difference on **every** apply. The symptom is
`kubectl apply` reporting `configured` forever instead of `unchanged`, and `kubectl diff` /
GitOps drift detection reporting phantom drift — verified on a live cluster. The renderer
rejects unquoted quantities in `config.yml` so the error names the user's own line.

Anything else the user wants to tune (concurrency limits, access logging, CORS, tracing,
metrics) goes in `config_overrides`, not in a Kubernetes field: point them at
`references/customization.md`, which splits the whole surface into "what you set in
config.yml", "`BENTOML_CONFIG_OVERRIDES` (i.e. `config_overrides`)", "looks tunable but is
not", and "BentoCloud-only, ignore".

Two rules behind `config_overrides`:
- The renderer nests what you write under the **named service** —
  `{"services":{"<ServiceName>":{...}}}`. A global `{"services":{"<key>":...}}` is merged
  *underneath* each service's decorator config and therefore loses to it. Write bare keys
  (`workers`, `traffic`, `logging`) and let the renderer key them.
- `<ServiceName>` is the BentoML service name verbatim (`TextPipeline`), never the slug —
  which is exactly why the `services:` blocks in `config.yml` are keyed by service name.

## Step 3 — Namespace and Secrets (imperative, before the config)

Create the namespace if it does not exist — **nothing downstream ever creates one**: the
deploy bundle refuses to run against a missing namespace, and it never renders a Namespace
object. (If the user wants the namespace tracked in git, apply
`templates/namespace.yaml` yourself; it is not part of the rendered set.)

```bash
kubectl --context <ctx> get namespace <ns> || kubectl --context <ctx> create namespace <ns>
```

**Application secrets** — one Secret per bento holding all sensitive env vars, shared by
every service that needs them (use per-service Secrets only if the user wants different
values per service). Ask the user to provide values (or confirm reading them from their
shell env); never echo values back, never put them in a file:

```bash
kubectl --context <ctx> create secret generic <bento-slug>-env -n <ns> \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  --from-literal=OTHER_KEY="..."
```

Then reference it by name: `services.<Name>.env_from_secrets: [<bento-slug>-env]`.

**Registry pull secret** (only for private registries) — one per namespace; every
service's Deployment references it, since all services run the same image:

```bash
kubectl --context <ctx> create secret docker-registry <bento-slug>-regcred -n <ns> \
  --docker-server=<registry> --docker-username=<user> \
  --docker-password="$REGISTRY_TOKEN"
```

→ `kubernetes.image_pull_secret: <bento-slug>-regcred`. Registry-specific details (GHCR,
ECR, Docker Hub, local kind/minikube): `references/private-registries.md`.

## Step 4 — Install the deploy bundle, then write `deploy/config.yml`

The manifests are rendered and the config is validated by **one implementation**, the
deploy bundle shipped with the sibling `bentoml-deploy-scriptgen` skill. This skill does
not carry a second renderer, and you must not write one: the interactive path and the CI
path share one renderer and one validator *by construction*, so what you review here is
exactly what CI applies later. Both skills ship together in the `bentoml-deploy` plugin, so
the sibling is normally present at `<skills-dir>/bentoml-deploy-scriptgen`.

Copy it in verbatim (the `__pycache__` cleanup is mandatory — running the templates in
place leaves caches inside the skill and `cp -R` drags them into the user's repo):

```bash
SCRIPTGEN=<path-to-bentoml-deploy-scriptgen-skill>
mkdir -p <project>/deploy
cp -R "$SCRIPTGEN/templates/deploy/." <project>/deploy/
find <project>/deploy -name __pycache__ -type d -prune -exec rm -rf {} +
find <project>/deploy -name '*.pyc' -delete
```

`deploy/deploy.py` needs **PyYAML** (`pip install pyyaml`; already a `bentoml` dependency)
— the same requirement Step 0 checked.

If `bentoml-deploy-scriptgen` is **not installed**, stop and get it rather than improvising
manifests: install the `bentoml-deploy` plugin (both skills come together), or fetch that
skill's `templates/deploy/` directory. A hand-rolled renderer is exactly the divergence
this design removes.

Then write `<project>/deploy/config.yml` — starting from `templates/config.yml` in *this*
skill, an annotated worked example of the schema — with the real topology and the Step 2
answers. The copy brought the bundle's own placeholder `config.yml`; overwrite it. The file
must sit next to `deploy.py`, which is where `deploy.py` looks for it by default (`--config
PATH` overrides that).

Rules for writing it:

1. **One `services:` block per service in `bento.yaml`**, keyed by the BentoML service
   name verbatim. No extra blocks, none missing. Exactly one has `entry: true`.
2. **Keep every key of the schema present**, even at its default (`env: {}`,
   `autoscaling.enabled: false`, …). This file is the user's map of what they can change;
   an omitted knob is an invisible one. **Two exceptions: `expose:` and `ingress:` appear
   ONLY under the entry service.** On a dependency they are rejected at load time (exit 2),
   because a non-entry service reachable from outside the cluster is an unauthenticated
   pickle endpoint. Do not add them "for completeness".
3. **Keep the comments.** They are the documentation for the file the user now owns, and
   the reason the file replaces the manifests as "the config". Adapt them to the actual
   topology; do not strip them to make the output shorter. Comments that only make sense
   for the template's example (e.g. its illustration of the both-pods timeout trap) can
   go, and per-service comments may be terser after the first block — the non-obvious
   facts must stay: the `readiness_timeout_seconds ≥ 6` floor and why, that decorator
   `resources` are inert in OSS so these values are the only ones that bind, that
   `config_overrides` beats the decorator only under the named service, that leaves need
   no `depends`, and that only the entry service may be exposed (pickle → RCE).
4. `project.dir` is the project root **resolved relative to `config.yml`** and must exist:
   `..` for the standard `deploy/config.yml` layout.
5. Quantities are **quoted strings**. `depends` lists **direct** dependencies only.
   Secrets are referenced by name, never inlined.

Then validate with the bundle's loader — **the only validator there is**:

```bash
grep -n '{{' <project>/deploy/config.yml && echo "ERROR: unreplaced placeholder" || echo "OK"
python3 <project>/deploy/deploy.py --target k8s --check-only --local-only
```

`--check-only --local-only` runs every check that needs no cluster or registry (config,
tooling) and is the same gate a CI PR job uses; drop `--local-only` once the context is
confirmed to also check cluster reachability, RBAC and the namespace. A config error exits
**2** with a path-precise message (e.g. `services.Gateway.expose.node_port: …`) and a hint.
It enforces: exactly one `entry: true`, every `depends` name exists, no cycles, slugs
DNS-1035 and unique, resource quantities quoted, `node_port` in range,
`readiness_timeout_seconds ≥ 6` (with the 5 s-fan-out reason), `expose`/`ingress` only on
the entry service, `registry_type: none` ⇒ `local_image_preloaded: true`, no
`BENTOML_RUNNER_MAP` in any `env`, and it **warns** on unknown keys — treat every warning
as a typo to fix, since an ignored key means a knob that silently does nothing.

Two warnings ARE expected here and must not be "fixed":
`working tree is dirty — the git-SHA image tag will not uniquely identify this build`
(you have not committed `config.yml` yet; it goes away once you do), and, on a project
without a git checkout, the missing-tag notice — both are why Step 5 passes `--image`
explicitly.

Also check by eye that the service keys match `bento.yaml`'s `services[].name` set exactly
— a typo there is not a validation error, it is a service that never gets deployed while a
phantom one does.

Tell the user to **commit `config.yml`**: it is the deployment config and contains no
secrets.

One handoff note for CI: this step pinned the image with `--image <reviewed ref>`, but an
agent-free `deploy.py --target k8s` run defaults the tag to the project's current short git
SHA. So CI's first run re-tags and restarts the pods once, reporting every Deployment
`configured`. That is correct — just not silent. Pass `--image` (or `--version`) in CI too
if you want the exact ref you reviewed here.

## Step 5 — Render the manifests and show them for review

Render into `<project>/k8s/` and show the user what will be applied. Nothing touches the
cluster in this step.

```bash
IMAGE=<full pushed image ref>          # from the containerize handoff
python3 <project>/deploy/deploy.py --target k8s --render-only k8s --image "$IMAGE"
```

**Always pass `--image` explicitly.** Without it the bundle derives the tag from the
project's short git SHA, which renders a tag that is not the image you just pushed (and
hard-fails outside a git checkout). `--render-only` writes the files, prints the rollout
order it would use, and exits without contacting the cluster; with no DIR it writes to
`deploy/rendered/`. It refuses to run while `kubernetes.manifests_dir` is set, since then
nothing is rendered at all.

The rendered set, per service, named by slug:

- `k8s/<slug>-deployment.yaml` and `k8s/<slug>-service.yaml` — always, one pair per service.
- `k8s/<slug>-hpa.yaml` — only for services with `autoscaling.enabled: true`.
- `k8s/ingress.yaml` — only when the entry service has `ingress.enabled: true`.

There is **no** rendered `namespace.yaml`: the namespace must already exist (Step 3).

For a single-service bento that is exactly two files, e.g. `k8s/summarization-deployment.yaml`
and `k8s/summarization-service.yaml`.

`templates/*.yaml` in this skill document the **shape** of each rendered file, field by
field, with the `config.yml` key behind each value. Read them when reviewing output or when
a user asks what a field is for; they are not edited to change a deployment, and they are
not what the renderer reads — the bundle renders from `config.yml` directly.

Sanity-check the rendered output. The bundle's own preflight (`--check-only`) already
covers the config side; these greps are the cheap read-back on what landed on disk, and the
dry-run is the API server's opinion of it:

```bash
# no leftover placeholders anywhere:
grep -n '{{' k8s/*.yaml && echo "ERROR: unreplaced placeholders" || echo "OK"
# the greps below skip comment lines:
grep -rn '^[^#]*command:' k8s/ && echo "ERROR: never override command:"
grep -rn '^[^#]*RUNNER_MAP' k8s/ && echo "ERROR: never set the runner map"
# every resource quantity must be a QUOTED string:
grep -hE '^\s+(cpu|memory|nvidia\.com/gpu):' k8s/*-deployment.yaml \
  | grep -v '"' && echo "ERROR: unquoted quantity" || echo "OK: all quantities quoted"
# every Service except the entry one must be ClusterIP:
grep -H 'type:' k8s/*-service.yaml
# dependency wiring: one pair per direct dependency, whitespace-separated, service names verbatim
grep -A1 -n 'BENTOML_SERVE_DEPENDS' k8s/*-deployment.yaml
# server-side, never apply:
kubectl --context <ctx> apply -n <ns> --dry-run=client -f k8s/
```

`k8s/` is **derived output**. Commit it only if the user wants reviewable diffs or GitOps;
if they do, say plainly that it must be re-rendered after every `config.yml` change, and
that the source of truth is still `config.yml`. Editing `k8s/` by hand is only correct
after deliberately switching to `kubernetes.manifests_dir` (Step 8).

## Step 6 — Confirm, then apply in the derived order

Show the user the rendered manifests plus a one-line summary —
**`context=<ctx>  namespace=<ns>  image=<image>  services=<slug1>,<slug2>(entry)`** — and
wait for explicit confirmation.

Then apply in the **derived rollout order** (the line `--render-only` printed:
dependencies first, entry service last), waiting for each rollout before the next. The
entry service's `/readyz` fans out to its dependencies by default (`runner_probe.enabled`),
so an entry pod started before its dependencies exist will sit un-ready and log connection
errors:

```bash
ROLLOUT=<kubernetes.rollout_timeout_seconds>   # 900 by default; NOT the startup budget
# order from --render-only's "rollout order would be:" line, deepest first, entry last:
for SLUG in sentiment text-pipeline; do
  kubectl --context <ctx> apply -n <ns> \
    -f k8s/$SLUG-deployment.yaml -f k8s/$SLUG-service.yaml
  [ -f k8s/$SLUG-hpa.yaml ] && kubectl --context <ctx> apply -n <ns> -f k8s/$SLUG-hpa.yaml
  kubectl --context <ctx> rollout status deployment/$SLUG -n <ns> --timeout=${ROLLOUT}s
done
[ -f k8s/ingress.yaml ] && kubectl --context <ctx> apply -n <ns> -f k8s/ingress.yaml
```

A single bulk apply (`kubectl apply -n <ns> -f k8s/`) also converges eventually — the
10-minute startupProbe budget covers the wait — which is what the generated deploy script
does; ordered apply gives clearer failures, so prefer it interactively.

**If apply partially fails** (some manifests created, one rejected — e.g. an invalid
Service name that dry-run did not catch): fix `config.yml`, re-render, then re-apply. Two
rules for the cleanup:

- Resources **this skill just created in this run** are yours to fix: deleting and
  re-applying them is allowed and is NOT covered by the "never delete" safety rule —
  that rule protects pre-existing resources you did not create.
- If the fix involves **renaming** (e.g. correcting a slug), `kubectl apply` would create a
  second object under the new name instead of updating the old one — first delete the
  just-created misnamed objects
  (`kubectl --context <ctx> delete -n <ns> deployment/<old> service/<old>`, only the ones
  created moments ago in this run), then change the slug in `config.yml`, re-render (which
  updates the callers' dependency URLs for you) and re-apply.

## Step 7 — Verify

1. **Rollout of every service**, in the same order. Use
   `kubernetes.rollout_timeout_seconds` (default 900), which is deliberately **longer** than
   a pod's startup budget (`probes.startup_failure_threshold` × 10 s = 600 s by default):
   with the two equal, a pod that legitimately uses its whole startup budget loses the race
   and the rollout is reported failed at the moment it succeeds.

```bash
for SLUG in sentiment text-pipeline; do
  kubectl --context <ctx> rollout status deployment/$SLUG -n <ns> --timeout=${ROLLOUT}s
done
kubectl --context <ctx> get pods -n <ns> -l app.kubernetes.io/part-of=<bento-name>
```

If a rollout stalls, inspect
(`kubectl --context <ctx> get pods -n <ns> -l app.kubernetes.io/name=<slug>`,
`kubectl --context <ctx> describe pod <pod> -n <ns>`,
`kubectl --context <ctx> logs <pod> -n <ns>`) and hand off to the
`bentoml-k8s-troubleshoot` skill for diagnosis. A dependency pod that is up but not
reachable makes the *caller* pod un-ready — check the dependency's Service endpoints
(`kubectl --context <ctx> get endpoints <dep-slug> -n <ns>`) and that the rendered
`BENTOML_SERVE_DEPENDS` hostname matches that Service name exactly.

2. **One real inference request against the ENTRY service only.** Do not port-forward to a
   dependency and do not try to call it with JSON — it speaks
   `application/vnd.bentoml+pickle` and will reject you. Exercising the entry endpoint is
   what proves the dependency wiring works.

   Derive the endpoint and payload from the entry service's API (each `@bentoml.api`
   method `def foo(self, text: str)` is `POST /foo` with JSON body `{"text": "..."}`);
   `bento.yaml`'s `schema.routes` lists them, or fetch
   `http://127.0.0.1:<local_port>/docs.json` while the port-forward is up, or ask the
   user. Record what you used as `verify.inference` in `config.yml`.
   If you need the schema first, run the block below once with only the readyz check
   plus a `curl -s .../docs.json`, then re-run it with the inference call filled in.

   Forward to the **uncommon local port from `kubernetes.local_port`** (e.g. 3130), not
   local 3000: port 3000 is commonly occupied by dev servers, and if the port-forward
   fails to bind, your curls silently hit whatever local process squats there — producing
   convincing but fake results. For the same reason the block checks that the port-forward
   process is still alive before trusting any curl.

   The whole sequence below — port-forward, checks, kill — **must run in ONE shell
   invocation**: the background process and `$PF_PID` do not survive across separate
   shell calls.

```bash
PORT=<kubernetes.local_port>
PF_ERR=$(mktemp)
kubectl --context <ctx> port-forward svc/<entry-slug> -n <ns> $PORT:3000 >"$PF_ERR" 2>&1 &
PF_PID=$!
OK=""
for i in $(seq 1 15); do
  kill -0 $PF_PID 2>/dev/null || { echo "port-forward exited:"; cat "$PF_ERR"; exit 1; }
  curl -sfo /dev/null http://127.0.0.1:$PORT/readyz && { OK=1; break; }
  sleep 2
done
[ -n "$OK" ] || { echo "not ready after 30s; port-forward log:"; cat "$PF_ERR"; kill $PF_PID; exit 1; }
echo READY

# Real inference request — the verify.inference path/body from config.yml:
curl -s -X POST http://127.0.0.1:$PORT/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text": "Kubernetes is an open-source container orchestration system."}'

kill $PF_PID
rm -f "$PF_ERR"
```

**Judge the inference response by its content, not the status code.** Only a correct
inference result proves the deployment serves traffic: HTTP 200 alone proves nothing, and a
4xx/5xx or error body means the payload or service needs fixing. If the output looks
unrelated to the service's API, suspect a local port squatter — confirm the port-forward
owns the port (`kill -0 $PF_PID`, check `$PF_ERR`) before believing anything.

3. **For a multi-service bento, prove the dependency ran in its OWN pod.** This is a
   separate check, and it is the one that catches the worst failure mode. A mis-wired
   dependency (a `depends` entry naming the wrong service, or missing entirely) does
   **not** produce an error: the caller instantiates the dependency in-process, so

   - `/readyz` returns **200** — the readiness fan-out only probes dependencies that are
     remote proxies, and an in-process dependency is simply not in that list, so a
     mis-wired pod looks *more* healthy, not less;
   - the inference response is **completely correct** — the same Python code ran, just in
     the wrong pod, loading every model into the caller.

   So neither readiness nor a correct answer is evidence. The only evidence is on the
   dependency's side. Run this immediately after the inference call:

```bash
# the dependency pod must show an access-log line for the call you just made:
kubectl --context <ctx> logs -n <ns> -l app.kubernetes.io/name=<dep-slug> --tail=20 --since=2m
# and it must have handled >0 requests (0 or a missing metric == it was never called).
# NOTE: bento images ship neither curl nor wget, so scrape /metrics with the
# interpreter that is always there:
kubectl --context <ctx> exec -n <ns> deploy/<dep-slug> -- python3 -c \
  "import urllib.request as u; print(''.join(l for l in u.urlopen('http://localhost:3000/metrics').read().decode().splitlines(True) if l.startswith('bentoml_service_request_total') and 'livez' not in l and 'readyz' not in l))"
```

A non-zero exit or empty output from that command is **not** proof of the fallback —
distinguish "the dependency served nothing" from "the command itself failed" before
concluding anything.

**In a DAG deeper than two tiers, a moving counter is not enough.** It proves the
dependency served *something*, not *who called it*. If a middle service's own `depends` is
empty and the leaf was instead wired into the entry service, the leaf's counter still
moves and answers stay correct. **Every non-leaf service needs its own `depends`.** The
discriminating evidence is the **client pod IP in the dependency's access log**: it must be
the pod of the service that declares `bentoml.depends()` on it, not the entry pod. Get the
IPs with

```bash
kubectl --context <ctx> get pods -n <ns> -o custom-columns='POD:.metadata.name,IP:.status.podIP' \
  -l app.kubernetes.io/part-of=<bento-name>
```

and check each dependency's log line against the caller you expect (e.g. a leaf reached
only through a middle tier must show the MIDDLE service's pod IP).

   No access-log line and no request counter on the dependency while the entry service
   returned a correct answer **is** the in-process fallback. Fix the `depends` entry in
   `config.yml` (the name must equal the BentoML service name exactly — including any
   `@bentoml.service(name=...)` override), re-render, re-apply; do not report success.

## Step 8 — Tell the user how to reach it, and how to change it

Access instructions apply to the **entry service** only (full details and commands in
`references/exposure-options.md`):

- **port-forward**: `kubectl --context <ctx> port-forward svc/<entry-slug> -n <ns> <local_port>:3000` → `http://127.0.0.1:<local_port>` (an uncommon local port, for the squatter reason in Step 7)
- **NodePort**: `http://<node-ip>:<node-port>` (get with `kubectl --context <ctx> get svc <entry-slug> -n <ns>` and `kubectl --context <ctx> get nodes -o wide`)
- **LoadBalancer**: external IP/hostname from `kubectl --context <ctx> get svc <entry-slug> -n <ns> -w`
- **Ingress**: `http(s)://<ingress.host>/` once DNS points at the ingress controller

Also mention: interactive API docs at `/` (Swagger UI), Prometheus metrics at `/metrics`
(per pod, so each service has its own), and — for a multi-service bento — that the
dependency Services are reachable only from inside the cluster, by design.

Then the change workflow, which is now one loop for everything:

```bash
# 1. edit deploy/config.yml   2. re-render   3. re-apply
python3 deploy/deploy.py --target k8s --render-only k8s --image "$IMAGE"
kubectl --context <ctx> apply -n <ns> -f k8s/      # or per-service, in rollout order
```

Or hand the whole loop to the bundle: `python3 deploy/deploy.py --target k8s
--skip-build --image "$IMAGE"` builds nothing, applies in rollout order, waits and verifies.

- **Workload shape** (replicas, resources, probes, node placement, exposure, autoscaling,
  dependency wiring): `config.yml` → re-render → apply.
- **Runtime behaviour** (timeouts, workers, concurrency limits, logging, CORS, tracing):
  `services.<Name>.config_overrides` in the same file — no image rebuild needed. See
  `references/customization.md`.
- **New image**: re-render with the new `$IMAGE` and apply; `config.yml` does not change
  (only `image.registry`/`repository` live there, not the tag).
- **Outgrowing the config** (sidecars, volumes, PDBs, affinity, `path_prefix` probe
  paths): render once, commit `k8s/`, set `kubernetes.manifests_dir: k8s` and take
  ownership. Say what is lost as well as gained — the config no longer shapes the
  workload, and the derived guarantees (rollout order, `BENTOML_SERVE_DEPENDS`,
  ClusterIP-only dependencies, HPA-vs-replicas) become the user's job. `templates/*.yaml`
  is the checklist of what those files must keep.
- **Production / CI-CD**: nothing to migrate — `deploy/` is already the committable
  bundle. Commit it (with `.gitignore` covering `__pycache__/` and `deploy/rendered/`) and
  run `deploy/deploy.py` from CI; use `bentoml-deploy-scriptgen` when the user wants its
  README, the EC2 target, or a regenerated bundle.

## Scope notes

Autoscaling is offered only as the plain HorizontalPodAutoscaler rendered from
`autoscaling.*` (CPU utilization out of the box; in-flight-request scaling if the cluster
has prometheus-adapter or KEDA). Scale-to-zero and canary/blue-green rollouts were
BentoCloud features and are **not** part of this skill — if the user needs them, say so
and stop rather than improvising.

## Files in this skill

- `templates/config.yml` — annotated example of the `bentoml-deploy-config/v3` schema; the starting point for the user's `deploy/config.yml`.
- `templates/{deployment,service,hpa,ingress}.yaml` — the documented shape of each rendered object (review reference, and the checklist for `manifests_dir` users). Not read by the renderer.
- `templates/namespace.yaml` — the optional Namespace object; nothing renders one, so apply it yourself if the user wants it in git.

Rendering and config validation live in the sibling `bentoml-deploy-scriptgen` skill's
`templates/deploy/` bundle, copied into the project in Step 4. There is deliberately no
second implementation here.

## References (read on demand)

- `references/customization.md` — the full "what can I customize" map: `config.yml` keys vs. `config_overrides` env knobs vs. BentoCloud-only keys, plus multi-service gotchas and HPA metric details.
- `references/exposure-options.md` — choosing and configuring port-forward / NodePort / LoadBalancer / Ingress, TLS, inference-friendly timeouts.
- `references/private-registries.md` — imagePullSecrets for GHCR/ECR/Docker Hub/self-hosted; kind/minikube local image loading; ttl.sh for throwaway tests.
- `references/gpu-scheduling.md` — verifying the NVIDIA device plugin, requesting `nvidia.com/gpu`, node selectors and taints.
