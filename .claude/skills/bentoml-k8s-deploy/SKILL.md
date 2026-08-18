---
name: bentoml-k8s-deploy
description: Deploy a containerized BentoML service to a vanilla Kubernetes cluster using plain kubectl manifests (no Helm, no operators, no BentoCloud/Yatai). Takes a pushed container image (built by the bentoml-containerize skill or `bentoml containerize`), discovers the bento's service topology, generates one Deployment + Service per BentoML service (plus optional HPA/Ingress) under k8s/, applies them in dependency order, and verifies the rollout with a real inference request. Use when the user says things like "deploy my BentoML service to Kubernetes", "deploy this bento image to my cluster", "run my bento on k8s", "create k8s manifests for my bento", "split my bento services into separate pods", or "expose my BentoML service in Kubernetes".
---

# Deploy a BentoML service to vanilla Kubernetes

You will take a **pushed container image reference** (e.g. `ghcr.io/acme/summarization:v1`,
produced by the `bentoml-containerize` skill), discover which BentoML services the bento
contains, generate plain Kubernetes manifests from the templates in this skill's
`templates/` directory, apply them with `kubectl`, and verify the deployment end to end.

**One Kubernetes Deployment per BentoML service.** A bento can package several
`@bentoml.service` classes wired together with `bentoml.depends()`; each becomes its own
Deployment + Service so it can be sized, scaled and scheduled independently. A
**single-service bento is the degenerate case** and works exactly the same way: one
Deployment, one Service, `--service-name` still passed, no dependency wiring. Do not
special-case it.

**The generated manifests ARE the deployment config file.** They are rendered once,
heavily commented, and then owned by the user: to change replicas, resources, probe
timings or autoscaling, they edit the YAML and re-apply. There is deliberately no
config-driven renderer and no second source of truth. Render them into the project
(`k8s/`), tell the user to commit them, and when they later ask to "change the config",
edit the manifest and re-apply rather than re-rendering from the template.

> For production / CI-CD deployments, generate a standalone script bundle (no agent needed
> at deploy time) with the `bentoml-deploy-scriptgen` skill. It consumes the exact same
> manifest layout.

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
  dependencies with `BENTOML_SERVE_DEPENDS` only.
- `@bentoml.service(resources={"cpu": ..., "memory": ...})` is **inert in open-source
  BentoML** — it constrains nothing at serve time. That is precisely why the declared
  values must be mirrored into the manifest's `requests`/`limits`.
- Models are usually baked into the image; models downloaded at runtime (e.g. gated HF
  models) need env vars such as `HF_TOKEN` via a Kubernetes Secret.

Safety rules (apply throughout):
- **Never apply anything before the user has confirmed the kubectl context AND namespace.**
- Never `kubectl delete` anything this skill did not create in this session.
- Never write secret values into files on disk — secrets are created only with
  `kubectl create secret ... --from-literal`.
- Never give a **non-entry** service a Service type other than `ClusterIP`: inter-service
  traffic is `application/vnd.bentoml+pickle`, i.e. unauthenticated pickle
  deserialization. Exposing it outside the cluster is remote code execution.

## Step 0 — Preflight checks

Run, and stop with a clear message if any check fails:

```bash
kubectl version --client        # kubectl installed?
kubectl config get-contexts     # list contexts; note the current one (*)
```

Then **ask the user which context to use — never assume the current context is the
intended cluster**, even if there is only one. Show them the list and get an explicit
answer. Use `--context <ctx>` on every subsequent kubectl command (do not silently
switch their current context with `kubectl config use-context`).

With the confirmed context, verify connectivity and permissions:

```bash
kubectl --context <ctx> get nodes -L kubernetes.io/arch   # cluster reachable? note node arch
kubectl --context <ctx> auth can-i create deployment -n <ns>   # run once the namespace is known (Step 3)
```

If the image was built on Apple Silicon and the `ARCH` column above shows `amd64`
nodes, warn the user now: the image must have been containerized with
`--opt platform=linux/amd64`, otherwise pods will crash with "exec format error".

## Step 1 — Discover the service topology

Everything downstream (how many Deployments, which one is the entry, how to wire the
dependency env var, what defaults to pre-fill) comes from the bento's own metadata. Do
this before asking the user anything about sizing.

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
| `name` | bento name → `app.kubernetes.io/part-of` label, secret name prefix |
| `entry_service` | the BentoML service name that receives external traffic; the only one that may be exposed and the only one verify targets |
| `services[].name` | one Deployment + Service per entry in this list |
| `services[].dependencies[].service` | the DAG edges — the value is the **BentoML service name** of the callee |
| `services[].config` | that service's declared `resources` / `workers` / `traffic` → pre-filled sizing defaults (Step 2) |

Note that `services[]` is **not** in dependency order (the entry service is often first).
Derive the rollout order yourself: topologically, deepest dependencies first, entry last.

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
  service), but the `BENTOML_SERVE_DEPENDS` key fails **silently** — an unmatched key
  falls back to instantiating the dependency in-process. Always grep for `name=` in every
  `@bentoml.service(...)` before deriving names;
- every `x = bentoml.depends(Other)` class attribute → a dependency edge to `Other`, whose
  service name is resolved the same way (follow `Other` to its own decorator's `name=`);
- `@bentoml.service(path_prefix="/x")` if present → that service's `/livez`, `/readyz`
  **and** `/metrics` all move under `/x`, so its manifest's probe paths (and any HPA
  scrape path) must be prefixed to match. Flag it; do not leave the probes at the root;
- the class named by the build target (`bentoml build service:TextPipeline`, or the
  `service:` key in `pyproject.toml`/`bentofile.yaml`) → the entry service. If that is
  ambiguous, the entry service is the one nothing else depends on; if several qualify,
  ask the user.

Then say plainly which topology you found, e.g.:

```
bento text_pipeline: 2 services
  TextPipeline (entry) -> depends on Sentiment
  Sentiment
Rollout order: Sentiment, then TextPipeline
```

For a wider or deeper DAG, report one line per service with its DIRECT dependencies and
mark the tiers, e.g.:

```
bento gateway: 4 services (entry_service: Gateway)
  Gateway   (entry) -> Enricher, Sentiment      [fan-out]
  Enricher          -> Tokenizer                [middle tier — carries its OWN wiring]
  Sentiment         -> (leaf)
  Tokenizer         -> (leaf, 2 hops from the entry)
Rollout order: Tokenizer, Sentiment, Enricher, Gateway
```

Several topological orders are usually valid (here `Sentiment, Tokenizer, …` is equally
correct). Pick one deterministically — **deepest tier first, then alphabetically within a
tier** — and then use that same order everywhere: the apply loop, the rollout waits, and
`targets.k8s.services` in a generated script bundle. Consistency is what matters.

### Slug (object name) rule — DNS-1035

Each service gets a `<slug>`: the BentoML service name snake_cased, then `_` → `-`.
`TextPipeline` → `text-pipeline`, `Sentiment` → `sentiment`, `TTSService` → `tts-service`.

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

- Validate every slug against `^[a-z]([-a-z0-9]*[a-z0-9])?$` before rendering.
- Slugs must be **unique** across the bento's services. If two collide, ask the user.
- Never derive the slug from the image repo basename (a ttl.sh UUID like `1f0d3c2a-…`
  starts with a digit and means nothing). Derive it from the BentoML service name.
- **`kubectl apply --dry-run=client` does NOT reliably catch DNS-1035 violations for
  Services** — a bad name passes dry-run, then the real apply rejects the Service while
  the Deployment succeeds, leaving a half-applied state (see Step 5). Validate the slugs
  yourself; never rely on dry-run for this.

## Step 2 — Gather deployment parameters (per service)

Ask once for the bento-wide parameters, then present a **per-service table with defaults
already pre-filled from each service's `config` in `bento.yaml`**, and let the user accept
everything in one go or edit individual cells.

Bento-wide:

| Parameter | Default | Notes |
|---|---|---|
| Image reference | (required) | Full pushed ref incl. registry + tag. If the user only ran `bentoml containerize`, the image may exist only locally — it must be pushed (or `kind load` / `minikube image load` for local clusters; see `references/private-registries.md`). |
| Namespace | `default` | Offer to create a dedicated one. All services of the bento go in the **same** namespace — dependency DNS names are rendered with it. |
| Secrets | none | Tokens/keys (e.g. `HF_TOKEN`). Step 3, never written to manifests. |
| Private registry? | detect/ask | If pulling needs auth → imagePullSecret, Step 3. See `references/private-registries.md`. |
| Exposure (entry service only) | `port-forward` | One of: port-forward (testing), NodePort, LoadBalancer, Ingress. Read `references/exposure-options.md` first; check what the cluster supports (`kubectl --context <ctx> get ingressclass`, cloud LB availability). Non-entry services are always ClusterIP — do not offer a choice. |

Per service (one row per `services[]` entry):

| Parameter | Default | Notes |
|---|---|---|
| Replicas | `1` | Scale each service independently; the bottleneck is rarely the entry service. |
| CPU request / limit | `config.resources.cpu` if declared, else `"500m"` / `"2"` | If the decorator declares `resources={"cpu": "1"}`, pre-fill request `"1"` and limit `"1"` (or `"2"` for burst) and **say why**: that declaration does nothing in OSS BentoML, so the manifest is the only place it takes effect. **Always quoted** — see the quantity-quoting rule below. |
| Memory request / limit | `config.resources.memory` if declared, else `"1Gi"` / `"4Gi"` | Size to the model: OOMKilled pods usually mean the limit is below the model's real footprint. |
| GPU | `config.resources.gpu` if declared, else none | `resources.gpu` is the one decorator resource key that does something in OSS (it sets `CUDA_VISIBLE_DEVICES`), so it must be matched by `nvidia.com/gpu` in the manifest or the pod gets no device. Read `references/gpu-scheduling.md` and confirm the cluster advertises the resource. |
| Workers | `config.workers` if declared, else `1` | Not a manifest field — it goes in `BENTOML_CONFIG_OVERRIDES` (Step 4). Interacts with CPU limit: N workers want ~N cores. |
| Timeout | `config.traffic.timeout` if declared, else `60` | Also not a manifest field. If you raise a **dependency's** timeout you must raise it in the **caller's** pod too — the client timeout is read from the calling pod's own config. See `references/customization.md`. |
| HPA | off | Opt-in, per service. If the user wants it, render `<slug>-hpa.yaml` and ask for min/max replicas and the target (CPU % or in-flight requests). See Step 4 and `references/customization.md`. |
| Env vars (plain) | none | Non-sensitive config only. |

**Probe-timeout floor (do not lower it).** The rendered `startupProbe` and
`readinessProbe` use `timeoutSeconds: 6`, and **6 is a floor, not a default to tune
down**. When a service's `/readyz` fans out to a `bentoml.depends()` dependency, BentoML
gives that dependency a **hard-coded 5-second** budget, so `/readyz` can legitimately take
just over 5 s. At `timeoutSeconds: 3` the kubelet hangs up first: a dependency that answers
in 4 s leaves the calling pod **permanently unready**, at startup and forever after, with
nothing in the pod's own logs to explain it. The 5 s is a framework constant, not a
tunable — `runner_probe.timeout` is inert on BentoML 1.2+. Raise the value for a deep
dependency chain; never lower it, and keep leaf services on the same value so that adding
a dependency later does not silently break them. `livenessProbe` stays at
`timeoutSeconds: 3` because `/livez` is pod-local and never cascades.

**Quantity-quoting rule (do not skip this).** Every resource quantity — `cpu`, `memory`,
`nvidia.com/gpu` — must be rendered **as a quoted string**: `cpu: "1"`, `cpu: "500m"`,
`memory: "4Gi"`, `nvidia.com/gpu: "1"`. An unquoted whole number (`cpu: 1`) is a YAML
integer; the API server stores the quantity as the string `"1"`, so the strategic-merge
patch sees a difference on **every** apply. The symptom is `kubectl apply` reporting
`configured` forever instead of `unchanged`, and `kubectl diff` / GitOps drift detection
reporting phantom drift — verified on a live cluster: unquoted → `configured` on every
run, quoted → `configured` once, then `unchanged`. Unit-suffixed values (`500m`, `1Gi`)
happen to be strings already, but quote them too so the rule has no exceptions. The
templates ship the quotes around the placeholders; keep them, and keep them when the user
later edits a value by hand.

Anything else the user wants to tune (concurrency limits, access logging, CORS, tracing,
metrics) is a container-env knob, not a manifest field: point them at
`references/customization.md`, which splits the whole surface into "manifest",
"`BENTOML_CONFIG_OVERRIDES`", "looks tunable but is not", and "BentoCloud-only, ignore".

Two rules when you write `BENTOML_CONFIG_OVERRIDES`:
- Always nest under the **named service** — `{"services":{"<ServiceName>":{...}}}`. A
  global `{"services":{"<key>":...}}` is merged *underneath* each service's decorator
  config and therefore loses to it.
- `<ServiceName>` is the BentoML service name verbatim (`TextPipeline`), never the slug.

## Step 3 — Namespace and Secrets (imperative, before manifests)

Create the namespace if it does not exist (or render `templates/namespace.yaml` into
`k8s/namespace.yaml` if the user wants it tracked in git):

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

**Registry pull secret** (only for private registries) — one per namespace; every
service's Deployment references it, since all services run the same image:

```bash
kubectl --context <ctx> create secret docker-registry <bento-slug>-regcred -n <ns> \
  --docker-server=<registry> --docker-username=<user> \
  --docker-password="$REGISTRY_TOKEN"
```

Registry-specific details (GHCR, ECR, Docker Hub, local kind/minikube):
`references/private-registries.md`.

## Step 4 — Render manifests into the project's `k8s/` directory

### File layout

One pair of files per BentoML service, named by slug:

- `k8s/<slug>-deployment.yaml` — always, one per service.
- `k8s/<slug>-service.yaml` — always, one per service.
- `k8s/<slug>-hpa.yaml` — only for services the user opted into autoscaling. Render it
  for reference if asked, but **do not apply it** unless the user explicitly said yes.
- `k8s/namespace.yaml` — only if the user wants the namespace in git.
- `k8s/ingress.yaml` — only for Ingress exposure; points at the **entry** service only.

For a single-service bento this is exactly two files, e.g. `k8s/summarization-deployment.yaml`
and `k8s/summarization-service.yaml`.

### Rendering procedure (per file)

1. **Render**: run sed on the file in this skill's `templates/` directory, substituting
   every placeholder you have a value for (including placeholders inside OPTIONAL blocks
   that apply), and write the output to `<project>/k8s/<name>.yaml`. Never modify the
   templates themselves.
2. **Prune**: edit the `k8s/` copy in place and **delete every `OPTIONAL` block that does
   not apply**, including its comment lines. Each block's comment states exactly how many
   lines to delete.
   - **The stated counts are relative to the template's own line numbering**, so if you
     prune by line number you must delete the blocks **bottom-up** (descending line
     order). Deleting top-down shifts every later block and silently mangles the file.
     Matching each block by its comment text and deleting N lines from there is equally
     safe and is the more robust option.
   - If you delete **every** entry under `env:`, also delete the 4-line `env:` header —
     the `# ---- environment ---` line, the two OPTIONAL-section comment lines, and the
     `env:` key itself. A dangling `env:` with no items is invalid.
3. **Keep the comments.** They are the documentation for the config file the user now
   owns. Do not strip them to make the output shorter.
4. **Check**: a finished manifest must contain **no `{{...}}`** (verified below). The
   literal `__DEPLOY_IMAGE__` is *not* a placeholder — it stays.

### The image sentinel

Every Deployment keeps `image: __DEPLOY_IMAGE__` verbatim. The real reference is
substituted **at apply time**, so the committed manifest never pins a build tag and the
same files work for the next image and for `bentoml-deploy-scriptgen`:

```bash
IMAGE=ghcr.io/acme/summarization:v1
for f in k8s/*.yaml; do echo '---'; sed "s|__DEPLOY_IMAGE__|$IMAGE|g" "$f"; done \
  | kubectl --context <ctx> apply -n <ns> -f -
```

(The `echo '---'` matters: concatenating YAML files without a document separator produces
one invalid document.)

### Placeholder reference (all templates)

| Placeholder | Used in | Meaning / example |
|---|---|---|
| `{{SLUG}}` | deployment, service, hpa | This service's DNS-1035 slug, e.g. `text-pipeline`. Object name, `app.kubernetes.io/name` label, and selector. |
| `{{BENTOML_SERVICE_NAME}}` | deployment, service, hpa | The BentoML service name **verbatim** from `bento.yaml`, e.g. `TextPipeline`. Used for `--service-name`, the `component` label, and config-override paths. Never slugged here. |
| `{{BENTO_NAME}}` | all | `name` from `bento.yaml`, e.g. `text_pipeline` → `app.kubernetes.io/part-of`. |
| `{{NAMESPACE}}` | all | Target namespace, e.g. `ml-services`. Same for every service. |
| `{{REPLICAS}}` | deployment | Integer, e.g. `1`. Delete the line if an HPA owns the Deployment. |
| `{{CPU_REQUEST}}` / `{{CPU_LIMIT}}` | deployment | e.g. `500m` / `2`. Always both. |
| `{{MEMORY_REQUEST}}` / `{{MEMORY_LIMIT}}` | deployment | e.g. `1Gi` / `4Gi`. Always both. |
| `{{SERVE_DEPENDS}}` | deployment (optional) | Dependency wiring, see below. Only on services that have dependencies. |
| `{{CONFIG_OVERRIDES}}` | deployment (optional) | Compact JSON for `BENTOML_CONFIG_OVERRIDES`, e.g. `{"services":{"TextPipeline":{"workers":2}}}`. |
| `{{IMAGE_PULL_SECRET}}` | deployment (optional) | docker-registry Secret name from Step 3. |
| `{{ENV_NAME}}` / `{{ENV_VALUE}}` | deployment (optional) | One plain env var; duplicate the pair per extra var. |
| `{{ENV_SECRET_NAME}}` | deployment (optional) | generic Secret name from Step 3. |
| `{{GPU_COUNT}}` | deployment (optional) | Integer GPUs per pod, e.g. `1`. |
| `{{NODE_SELECTOR_KEY}}` / `{{NODE_SELECTOR_VALUE}}` | deployment (optional) | e.g. `nvidia.com/gpu.present` / `true`. |
| `{{SERVICE_TYPE}}` | service | `ClusterIP` for every non-entry service, always. Entry service: `ClusterIP` (port-forward / Ingress), `NodePort`, or `LoadBalancer`. |
| `{{NODE_PORT}}` | service (optional) | NodePort only, 30000–32767. |
| `{{SERVICE_ANNOTATION_KEY}}` / `{{SERVICE_ANNOTATION_VALUE}}` | service (optional) | Cloud LB tuning. |
| `{{MIN_REPLICAS}}` / `{{MAX_REPLICAS}}` | hpa | e.g. `1` / `5`. |
| `{{CPU_TARGET_PERCENT}}` | hpa (variant A) | e.g. `70`. |
| `{{ENTRY_SLUG}}` / `{{ENTRY_SERVICE_NAME}}` | ingress | The entry service's slug / BentoML name. |
| `{{INGRESS_CLASS_NAME}}` | ingress | From `kubectl --context <ctx> get ingressclass`, e.g. `nginx`. |
| `{{INGRESS_HOST}}` | ingress | e.g. `summarization.example.com`. |
| `{{INGRESS_ANNOTATION_KEY}}` / `{{INGRESS_ANNOTATION_VALUE}}` | ingress (optional) | e.g. nginx proxy timeouts. |
| `{{TLS_SECRET_NAME}}` | ingress (optional) | Existing TLS Secret. |

### Dependency wiring — `BENTOML_SERVE_DEPENDS`

For **every service that has dependencies** (in a two-tier bento that is only the entry
service; in a deeper DAG it is every non-leaf service), render the `BENTOML_SERVE_DEPENDS`
block in its Deployment. For every leaf service, delete the block.

The value is a **whitespace-separated** list of `ServiceName=URL` pairs — one per
**direct** dependency, using in-cluster DNS:

```
<DependencyServiceName>=http://<dependency-slug>.<namespace>.svc.cluster.local:3000
```

Rules that will silently break the deployment if ignored:
- `ServiceName` must be the **BentoML service name exactly** as in `bento.yaml`
  (`Sentiment`, not `sentiment`) — the lookup is by that key, and a miss means the
  dependency is instantiated in-process instead.
- The URL must contain **no `=`** (no query strings). Each pair goes through
  `split("=", maxsplit=2)` and then into `dict()`, so a second `=` anywhere produces three
  parts and raises a ValueError at startup.
- Pairs are separated by **whitespace**, not commas.
- Only **direct** dependencies. Transitive ones are wired in their own caller's pod.
- Do **not** also set `BENTOML_RUNNER_MAP`.

Example for `TextPipeline` depending on `Sentiment` in namespace `demos`:

```yaml
            - name: BENTOML_SERVE_DEPENDS
              value: "Sentiment=http://sentiment.demos.svc.cluster.local:3000"
```

### Example render

```bash
mkdir -p k8s
SKILL=<path-to-this-skill>
sed -e 's|{{SLUG}}|sentiment|g' \
    -e 's|{{BENTOML_SERVICE_NAME}}|Sentiment|g' \
    -e 's|{{BENTO_NAME}}|text_pipeline|g' \
    -e 's|{{NAMESPACE}}|demos|g' \
    -e 's|{{REPLICAS}}|1|g' \
    -e 's|{{CPU_REQUEST}}|1|g'    -e 's|{{CPU_LIMIT}}|2|g' \
    -e 's|{{MEMORY_REQUEST}}|1Gi|g' -e 's|{{MEMORY_LIMIT}}|4Gi|g' \
    "$SKILL/templates/deployment.yaml" > k8s/sentiment-deployment.yaml
```

Substitute the quantities **bare**, as above: the template already wraps those four
placeholders in quotes, so this renders `cpu: "1"` / `memory: "1Gi"`. Do not add a second
pair of quotes, and do not strip the template's.

Then prune: for a leaf service on a public registry with no secrets, no GPU and no node
pinning, delete the imagePullSecrets, `BENTOML_SERVE_DEPENDS`, `BENTOML_CONFIG_OVERRIDES`,
plain-env, envFrom, GPU and nodeSelector blocks — and the now-empty `env:` key. Repeat for
each service with its own values.

### Sanity checks (all of these, before showing anything to the user)

```bash
grep -n '{{' k8s/*.yaml && echo "ERROR: unreplaced placeholders" || echo "OK: no placeholders"
grep -l 'image: __DEPLOY_IMAGE__' k8s/*-deployment.yaml   # must list EVERY deployment file
# the greps below skip comment lines (the templates mention these on purpose):
grep -rn '^[^#]*command:' k8s/ && echo "ERROR: never override command:"
grep -rn '^[^#]*RUNNER_MAP' k8s/ && echo "ERROR: never set the runner map"
# every resource quantity must be a QUOTED string (see the quantity-quoting rule):
grep -hE '^\s+(cpu|memory|nvidia\.com/gpu):' k8s/*-deployment.yaml \
  | grep -v '"' && echo "ERROR: unquoted quantity -> every apply will report 'configured'" \
  || echo "OK: all quantities quoted"
# every Service except the entry one must be ClusterIP:
grep -H 'type:' k8s/*-service.yaml
# selector/label coherence: for each <slug>, name label == selector == object name
for f in k8s/*-deployment.yaml; do echo "== $f"; grep -n 'app.kubernetes.io/name' "$f"; done
# dry-run with the sentinel substituted (never apply):
for f in k8s/*.yaml; do echo '---'; sed "s|__DEPLOY_IMAGE__|$IMAGE|g" "$f"; done \
  | kubectl --context <ctx> apply -n <ns> --dry-run=client -f -
```

Committing `k8s/` is the point — it is the deployment config and contains no secrets. Tell
the user to commit it, and that editing these files plus re-applying is how they change
the deployment from now on.

## Step 5 — Confirm, then apply in dependency order

Show the user the rendered manifests plus a one-line summary —
**`context=<ctx>  namespace=<ns>  image=<image>  services=<slug1>,<slug2>(entry)`** — and
wait for explicit confirmation.

Then apply **dependencies first, entry service last**, waiting for each tier's rollout
before the next. The entry service's `/readyz` fans out to its dependencies by default
(`runner_probe.enabled`), so an entry pod started before its dependencies exist will sit
un-ready and log connection errors:

```bash
IMAGE=<image>
# tier by tier, deepest dependencies first, entry service last:
for SLUG in sentiment text-pipeline; do
  for f in k8s/$SLUG-deployment.yaml k8s/$SLUG-service.yaml; do echo '---'; sed "s|__DEPLOY_IMAGE__|$IMAGE|g" "$f"; done \
    | kubectl --context <ctx> apply -n <ns> -f -
  kubectl --context <ctx> rollout status deployment/$SLUG -n <ns> --timeout=600s
done
```

Apply `k8s/ingress.yaml` (if any) after the entry service. Do **not** apply
`k8s/*-hpa.yaml` unless the user opted in — and when you do, first delete `spec.replicas`
from that service's Deployment, or every apply will fight the HPA.

A single bulk apply of the whole directory also converges eventually (the 10-minute
startupProbe budget covers the wait), which is what the generated deploy script does — but
ordered apply gives clearer failures, so prefer it interactively.

**If apply partially fails** (some manifests created, one rejected — e.g. an invalid
Service name that dry-run did not catch): fix the offending manifest, then re-apply. Two
rules for the cleanup:

- Resources **this skill just created in this run** are yours to fix: deleting and
  re-applying them is allowed and is NOT covered by the "never delete" safety rule —
  that rule protects pre-existing resources you did not create.
- If the fix involves **renaming** (e.g. correcting a slug), `kubectl apply` would create a
  second object under the new name instead of updating the old one — first delete the
  just-created misnamed objects
  (`kubectl --context <ctx> delete -n <ns> deployment/<old> service/<old>`, only the ones
  created moments ago in this run), update the slug in **all** manifests that mention it
  (including the `BENTOML_SERVE_DEPENDS` URL in its caller's Deployment), then re-apply.

## Step 6 — Verify

1. **Rollout of every service**, in the same order (budget matches the startupProbe: up to
   ~10 min for model loading):

```bash
for SLUG in sentiment text-pipeline; do
  kubectl --context <ctx> rollout status deployment/$SLUG -n <ns> --timeout=600s
done
kubectl --context <ctx> get pods -n <ns> -l app.kubernetes.io/part-of=<bento-name>
```

If a rollout stalls, inspect
(`kubectl --context <ctx> get pods -n <ns> -l app.kubernetes.io/name=<slug>`,
`kubectl --context <ctx> describe pod <pod> -n <ns>`,
`kubectl --context <ctx> logs <pod> -n <ns>`) and hand off to the
`bentoml-k8s-troubleshoot` skill for diagnosis. A dependency pod that is up but not
reachable makes the *entry* pod un-ready — check the dependency's Service endpoints
(`kubectl --context <ctx> get endpoints <dep-slug> -n <ns>`) and that the
`BENTOML_SERVE_DEPENDS` hostname matches that Service name exactly.

2. **One real inference request against the ENTRY service only.** Do not port-forward to a
   dependency and do not try to call it with JSON — it speaks
   `application/vnd.bentoml+pickle` and will reject you. Exercising the entry endpoint is
   what proves the dependency wiring works.

   Derive the inference endpoint and payload from the entry service's API (each
   `@bentoml.api` method `def foo(self, text: str)` is `POST /foo` with JSON body
   `{"text": "..."}`); `bento.yaml`'s `schema.routes` lists them, or fetch
   `http://127.0.0.1:3100/docs.json` while the port-forward is up, or ask the user.
   If you need the schema first, run the block below once with only the readyz check
   plus a `curl -s http://127.0.0.1:3100/docs.json`, then re-run it with the inference
   call filled in.

   Forward to an **uncommon local port (3100)**, not local 3000: port 3000 is commonly
   occupied by dev servers, and if the port-forward fails to bind, your curls silently
   hit whatever local process squats there — producing convincing but fake results.
   For the same reason the block checks that the port-forward process is still alive
   before trusting any curl.

   The whole sequence below — port-forward, checks, kill — **must run in ONE shell
   invocation**: the background process and `$PF_PID` do not survive across separate
   shell calls.

```bash
PF_ERR=$(mktemp)
kubectl --context <ctx> port-forward svc/<entry-slug> -n <ns> 3100:3000 >"$PF_ERR" 2>&1 &
PF_PID=$!
OK=""
for i in $(seq 1 15); do
  kill -0 $PF_PID 2>/dev/null || { echo "port-forward exited:"; cat "$PF_ERR"; exit 1; }
  curl -sfo /dev/null http://127.0.0.1:3100/readyz && { OK=1; break; }
  sleep 2
done
[ -n "$OK" ] || { echo "not ready after 30s; port-forward log:"; cat "$PF_ERR"; kill $PF_PID; exit 1; }
echo READY

# Real inference request — substitute the endpoint/payload derived above:
curl -s -X POST http://127.0.0.1:3100/analyze \
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
   `BENTOML_SERVE_DEPENDS` (wrong key, wrong host, missing entirely) does **not** produce
   an error: the entry service instantiates the dependency in-process, so

   - `/readyz` returns **200** — the readiness fan-out only probes dependencies that are
     remote proxies, and an in-process dependency is simply not in that list, so a
     mis-wired pod looks *more* healthy, not less;
   - the inference response is **completely correct** — the same Python code ran, just in
     the wrong pod, loading every model into the gateway.

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
dependency served *something*, not *who called it*. If a middle service's own
`BENTOML_SERVE_DEPENDS` is missing and the leaf was instead wired into the entry
service, the leaf's counter still moves and answers stay correct. The discriminating
evidence is the **client pod IP in the dependency's access log**: it must be the pod of
the service that declares `bentoml.depends()` on it, not the entry pod. Get the IPs with

```bash
kubectl --context <ctx> get pods -n <ns> -o custom-columns='POD:.metadata.name,IP:.status.podIP' \
  -l app.kubernetes.io/part-of=<bento-name>
```

and check each dependency's log line against the caller you expect (e.g. a leaf reached
only through a middle tier must show the MIDDLE service's pod IP).

   No access-log line and no request counter on the dependency while the entry service
   returned a correct answer **is** the in-process fallback. Fix the
   `BENTOML_SERVE_DEPENDS` key (it must equal the BentoML service name exactly — including
   any `@bentoml.service(name=...)` override) and re-apply; do not report success.

## Step 7 — Tell the user how to reach the service

Access instructions apply to the **entry service** only (full details and commands in
`references/exposure-options.md`):

- **port-forward**: `kubectl --context <ctx> port-forward svc/<entry-slug> -n <ns> 3100:3000` → `http://127.0.0.1:3100` (an uncommon local port, for the squatter reason in Step 6)
- **NodePort**: `http://<node-ip>:<node-port>` (get with `kubectl --context <ctx> get svc <entry-slug> -n <ns>` and `kubectl --context <ctx> get nodes -o wide`)
- **LoadBalancer**: external IP/hostname from `kubectl --context <ctx> get svc <entry-slug> -n <ns> -w`
- **Ingress**: `http(s)://<INGRESS_HOST>/` once DNS points at the ingress controller

Also mention: interactive API docs at `/` (Swagger UI), Prometheus metrics at `/metrics`
(per pod, so each service has its own), and — for a multi-service bento — that the
dependency Services are reachable only from inside the cluster, by design.

Finally, tell them how to change things from here:
- **Workload shape** (replicas, resources, probes, node placement, exposure): edit
  `k8s/<slug>-*.yaml` and re-apply. Those files are the config.
- **Runtime behaviour** (timeouts, workers, concurrency limits, logging, CORS, tracing):
  add/adjust `BENTOML_CONFIG_OVERRIDES` in the relevant Deployment — no image rebuild
  needed. See `references/customization.md`.
- **New image**: nothing to re-render; re-run the apply with the new `$IMAGE` substituted
  for `__DEPLOY_IMAGE__`.

## Scope notes

Autoscaling is offered only as the plain HorizontalPodAutoscaler in
`templates/hpa.yaml` (CPU utilization out of the box; in-flight-request scaling if the
cluster has prometheus-adapter or KEDA). Scale-to-zero and canary/blue-green rollouts were
BentoCloud features and are **not** part of this skill — if the user needs them, say so
and stop rather than improvising.

## References (read on demand)

- `references/customization.md` — the full "what can I customize" map: manifest knobs vs. `BENTOML_CONFIG_OVERRIDES` env knobs vs. BentoCloud-only keys, plus multi-service gotchas and HPA metric details.
- `references/exposure-options.md` — choosing and configuring port-forward / NodePort / LoadBalancer / Ingress, TLS, inference-friendly timeouts.
- `references/private-registries.md` — imagePullSecrets for GHCR/ECR/Docker Hub/self-hosted; kind/minikube local image loading; ttl.sh for throwaway tests.
- `references/gpu-scheduling.md` — verifying the NVIDIA device plugin, requesting `nvidia.com/gpu`, node selectors and taints.
