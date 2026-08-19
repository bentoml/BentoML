---
name: bentoml-k8s-deploy
description: Deploy a containerized BentoML service to a vanilla Kubernetes cluster using plain kubectl manifests (no Helm, no operators, no BentoCloud/Yatai). Takes a pushed container image (built by the bentoml-containerize skill or `bentoml containerize`), discovers the bento's service topology, writes one `config.yml` for the deployment, renders one Deployment + Service per BentoML service (plus optional HPA/Ingress) from it, applies them in dependency order, and verifies the rollout with a real inference request. Use when the user says things like "deploy my BentoML service to Kubernetes", "deploy this bento image to my cluster", "run my bento on k8s", "create k8s manifests for my bento", "split my bento services into separate pods", or "expose my BentoML service in Kubernetes".
license: Apache-2.0
compatibility: >-
  Requires kubectl with access to a Kubernetes cluster, Python >= 3.9 with PyYAML (the bundled renderer runs locally), and an image registry the cluster can pull from; AWS CLI v2 for ECR.
---

# Deploy a BentoML service to vanilla Kubernetes

You will take a **pushed container image reference** (e.g. `ghcr.io/acme/summarization:v1`,
produced by the `bentoml-containerize` skill), discover which BentoML services the bento
contains, write a single **`config.yml`** into the project, **render** plain Kubernetes
manifests from it, apply them with `kubectl`, and verify the deployment end to end.

**`deploy/config.yml` is the deployment config — the one file the user edits.** It holds
the image repository, the cluster/namespace, and *optionally* per-service overrides
(replicas, resources, probes, env, autoscaling, exposure). The manifests under `k8s/` are
**rendered output**: derived, regenerable, not hand-owned. "Change the deployment" means
*edit `config.yml`, re-render, re-apply* — never *edit the YAML under `k8s/`*.

**The config does NOT record the topology.** The service list, which service is the entry
service, and the dependency DAG all come from the bento's own `bento.yaml`, which
`deploy.py` re-reads on every run. There is no `entry:` key and no `depends:` key. So a
config can never disagree with the code: adding a `bentoml.depends()` and rebuilding is
enough, and there is nothing here to keep in sync. A config that customizes nothing is
five lines (`templates/config.minimal.yml`).

**One Kubernetes Deployment per BentoML service.** A bento can package several
`@bentoml.service` classes wired together with `bentoml.depends()`; each becomes its own
Deployment + Service so it can be sized, scaled and scheduled independently. A
**single-service bento is the degenerate case** and works exactly the same way: one
Deployment, one Service, `--service-name` still passed, no dependency wiring. Do not
special-case it.

Derived, never asked of the user and never hand-maintained anywhere:

| Derived | From |
|---|---|
| The service list, the entry service, the DAG | `bento.yaml` (`services[].name`, `entry_service`, `services[].dependencies[].service`) |
| Rollout order | topological sort of that DAG — deepest tier first, alphabetical within a tier |
| `BENTOML_SERVE_DEPENDS` | the DAG + each dependency's slug + `kubernetes.namespace` |
| Slugs, labels and selectors | each service's name |
| ClusterIP on every non-entry Service | `bento.yaml`'s `entry_service` (an RCE invariant, see below) |
| The image tag | the bento version (by default the short git SHA `bentoml build --version` stamped) |
| The build platform | local builder arch vs. the cluster's node arch; a cross-build gets `--opt platform=linux/<node arch>` automatically, and says so |
| Registry auth | the `image` URL's host (`*.dkr.ecr.<region>.amazonaws.com` ⇒ ECR login + describe-or-create) |
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
  dependencies with `BENTOML_SERVE_DEPENDS` only, which the renderer derives from the DAG.
- `@bentoml.service(resources={"cpu": ..., "memory": ...})` is **inert in open-source
  BentoML** — it constrains nothing at serve time. That is precisely why the resource
  values (the defaults, or the ones you write in `config.yml`) are load-bearing: they are
  the only thing that constrains the pod.
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

The `ARCH` column matters for a pre-built image only. **There is no platform knob:** when
the bundle builds, it compares the local builder arch with the node arch and cross-builds
automatically. But an image someone already built on Apple Silicon for `amd64` nodes will
still crash-loop with "exec format error" — if the arches differ and the image was not
built with `--opt platform=linux/amd64`, say so now and rebuild rather than deploy.

The user also needs **a writable image registry the cluster can pull from** (or a local
cluster where the image is loaded onto the nodes — see `references/private-registries.md`).

## Step 1 — Discover the service topology

Do this before asking the user anything about sizing. Two things come out of it, and only
one of them ends up in a file:

1. **The report to the user** — what the bento actually contains, which service is the
   entry, how deep the DAG is. That framing is what makes the rest of the conversation
   possible.
2. **Sizing suggestions** — each service's declared `resources`/`workers`/`traffic`, used
   to pre-fill the Step 2 table.

**None of it is written into `config.yml`.** `deploy.py` rediscovers the topology from
`bento.yaml` on every run (from the freshly built bento, or out of the image with
`--skip-build`, or from the `deploy/.bento-topology.json` cache a previous run wrote), and
derives the rollout order, the wiring, the slugs and the labels from that. Say this to the
user in as many words: the shape of their bento is not duplicated anywhere, so it cannot
drift.

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
or `nerdctl`). If the bento is in the local store instead, `bentoml get <tag> -o json`
(or the `bento.yaml` at its store path) has the same content and needs no docker.

Read these fields:

| Field | Use |
|---|---|
| `name` | bento name → the `app.kubernetes.io/part-of` label, secret name prefix, and how you `kubectl get -l` the whole bento |
| `entry_service` | the BentoML service name that receives external traffic — the only one that may be exposed, the only one verify targets, and the only one whose `services:` block may carry `expose`/`ingress` |
| `services[].name` | the service set. These names are the **only valid keys** under `services:` in the config (verbatim) |
| `services[].dependencies[].service` | the DAG edges (the value is the **BentoML service name** of the callee). Report them; do not write them anywhere |
| `services[].config` | that service's declared `resources` / `workers` / `traffic` → pre-filled suggestions (Step 2) |

`services[]` is **not** in dependency order (the entry service is often first). Report the
derived order so the user can sanity-check the DAG; nothing records it.

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
  instantiating the dependency in-process. It is also the name a `services:` override
  block must use, and a key that matches no service is a load-time error. Always grep for
  `name=` in every `@bentoml.service(...)` before deriving names;
- every `x = bentoml.depends(Other)` class attribute → an edge to `Other`, whose service
  name is resolved the same way (follow `Other` to its own decorator's `name=`);
- `@bentoml.service(path_prefix="/x")` if present → that service's `/livez`, `/readyz`
  **and** `/metrics` all move under `/x`, so the rendered probe paths (and any HPA
  scrape path) must be prefixed to match. **The renderer has no config knob for this** —
  flag it to the user and treat that service as a `manifests_dir` case, or drop the
  prefix; do not leave the probes at the root;
- the class named by the build target (`bentoml build service:TextPipeline`, or the
  `service:` key in `pyproject.toml`/`bentofile.yaml`) → the entry service. If that is
  ambiguous, the entry service is the one nothing else depends on; if several qualify,
  ask the user — and record the build target as `build_target` in the config so the
  bundle's build stage resolves the same one.

Then say plainly which topology you found, e.g.:

```
bento text_pipeline: 2 services
  TextPipeline (entry) -> depends on Sentiment
  Sentiment
Rollout order (derived at deploy time): Sentiment, then TextPipeline
```

For a wider or deeper DAG, report one line per service with its DIRECT dependencies and
mark the tiers, e.g.:

```
bento gateway: 4 services (entry_service: Gateway)
  Gateway   (entry) -> Enricher, Sentiment      [fan-out]
  Enricher          -> Tokenizer                [middle tier]
  Sentiment         -> (leaf)
  Tokenizer         -> (leaf, 2 hops from the entry)
Rollout order (derived at deploy time): Sentiment, Tokenizer, Enricher, Gateway
```

Several topological orders are valid; the derived one is **deepest tier first, then
alphabetically within a tier** (hence `Sentiment` before `Tokenizer` above).

### Slug (object name) rule — DNS-1035

Each service gets a `<slug>`: the BentoML service name snake_cased, then `_` → `-`.
`TextPipeline` → `text-pipeline`, `Sentiment` → `sentiment`, `TTSService` → `tts-service`.
The renderer derives it; `services.<Name>.slug` exists only to override it (a collision, or
a legacy object name to keep). Compute it while reporting the topology so the user sees the
object names they are about to get:

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
  exits 2 naming `services.<Name>.slug`. A service name starting with a digit or made of
  non-ASCII characters cannot produce a valid slug — set one explicitly.
- Slugs must be **unique** across the bento's services. If two derived slugs collide, ask
  the user which one to rename, and write that `slug` override.
- Never derive the slug from the image repo basename (a ttl.sh UUID like `1f0d3c2a-…`
  starts with a digit and means nothing). It always comes from the BentoML service name.
- **`kubectl apply --dry-run=client` does NOT reliably catch DNS-1035 violations for
  Services** — a bad name passes dry-run, then the real apply rejects the Service while
  the Deployment succeeds, leaving a half-applied state (see Step 6). Never rely on
  dry-run for this.

## Step 2 — One round of questions

Ask everything in **one round**, and lead with the fact that **every question has a
working default**: the user can say "all defaults" and get a five-line config. Then write
`config.yml` (Step 4) — from that point on, further changes are config edits, not new
questions.

Bento-wide:

| Parameter | Config key | Default | Notes |
|---|---|---|---|
| Image repository | `image` | (required) | **One URL, no tag** — e.g. `303081928216.dkr.ecr.us-west-1.amazonaws.com/bml-demo`. The tag is the bento version, so it is not in the config; registry auth is derived from the host (ECR is detected and logged into). If the image exists only locally and the cluster is kind/minikube, load it onto the nodes and set `image: ""` (see `references/private-registries.md`). |
| Namespace | `kubernetes.namespace` | `default` | Offer to create a dedicated one. All services of the bento go in the **same** namespace — dependency DNS names are rendered with it. |
| Project root | `project` | `..` | A scalar path, resolved relative to `config.yml`; `..` for the standard `deploy/config.yml` layout. |
| Build target | `build_target` | omitted | Only when plain `bentoml build` cannot resolve which service to build. |
| Secrets | `services.<Name>.env_from_secrets` | none | Tokens/keys (e.g. `HF_TOKEN`). Created in Step 3; only the Secret **name** ever reaches the config. |
| Private registry? | `kubernetes.image_pull_secret` | detect/ask | If pulling needs auth → imagePullSecret, Step 3. See `references/private-registries.md`. |
| Exposure (entry service only) | `services.<Entry>.expose` / `.ingress` | port-forward (`ClusterIP`) | One of: port-forward, NodePort, LoadBalancer, Ingress. Read `references/exposure-options.md` first; check what the cluster supports (`kubectl --context <ctx> get ingressclass`, cloud LB availability). Non-entry services are always ClusterIP — do not offer a choice. |
| Verification call | `verify.inference` | omitted | Optional. With it, verification proves the model answers *and* that dependencies were called in their own pods. Without it, verification is `/readyz` only and the dependency-metrics proof is skipped with a logged note — worth taking two minutes to fill in for a multi-service bento. |

Per service (one row per `services[]` entry) — every one of these is an **override**; omit
the key, or the whole block, and the default applies:

| Parameter | Config key | Default | Notes |
|---|---|---|---|
| Replicas | `replicas` | `1` | Scale each service independently; the bottleneck is rarely the entry service. |
| CPU request / limit | `resources.requests.cpu` / `resources.limits.cpu` | `"500m"` / `"2"` | If the decorator declares `resources={"cpu": "1"}`, suggest request `"1"` and limit `"1"` (or `"2"` for burst) and **say why**: that declaration does nothing in OSS BentoML, so the config is the only place it takes effect. **Always quoted strings** — see the quoting rule below. |
| Memory request / limit | `resources.requests.memory` / `resources.limits.memory` | `"1Gi"` / `"4Gi"` | Size to the model: OOMKilled pods usually mean the limit is below the model's real footprint. Mirror `config.resources.memory` from `bento.yaml` when it is declared. |
| GPU | `resources.limits["nvidia.com/gpu"]` | none | `resources.gpu` is the one decorator resource key that does something in OSS (it sets `CUDA_VISIBLE_DEVICES`), so a service that declares it must request the device here or the pod gets none. Read `references/gpu-scheduling.md` and confirm the cluster advertises the resource; pair with `node_selector`/`tolerations`. |
| Workers | `config_overrides.workers` | `1` (the decorator's value applies if declared) | Server-side knob, rendered into `BENTOML_CONFIG_OVERRIDES`. Interacts with the CPU limit: N workers want ~N cores. |
| Timeout | `config_overrides.traffic.timeout` | `60` (the decorator's value applies if declared) | Also server-side. Raising a **dependency's** timeout is the one case a per-service block cannot express — see `references/customization.md`. |
| Autoscaling | `autoscaling.*` | `enabled: false` | Opt-in, per service; ask for min/max and the target (CPU % or in-flight requests per pod). The renderer then omits `spec.replicas` for that service. See `references/customization.md`. |
| Env vars (plain) | `env` | none | Non-sensitive config only. |
| Node placement | `node_selector`, `tolerations` | none | GPU pools, arch pinning, tainted node groups. |
| Object name | `slug` | derived from the service name | Only to resolve a collision or keep a legacy name. |

Note the asymmetry worth stating out loud: a value **declared in the decorator** still
governs the BentoML server (`workers`, `traffic.timeout`), but a declared
`resources.cpu`/`memory` governs **nothing** — so those two are the ones you actively
suggest mirroring into the config.

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
exactly what CI applies later. The five skills install together, whichever host you run
(`~/.claude/skills/`, `~/.codex/skills/`, a project `.codex/skills/`, the `bentoml-deploy`
plugin, …), so the sibling is normally the directory next to this one:
`<skills-dir>/bentoml-deploy-scriptgen`. If it is genuinely missing, say so and stop —
do not hand-write a renderer.

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
manifests — reinstall the set (`npx skills add bentoml/BentoML`, the `bentoml-deploy`
plugin, or a copy of the repo's `skills/` directory), or fetch just that skill's
`templates/deploy/` directory. A hand-rolled renderer is exactly the divergence this
design removes.

Then write `<project>/deploy/config.yml`, overwriting the placeholder the copy brought. It
must sit next to `deploy.py`, which is where `deploy.py` looks for it by default
(`--config PATH` overrides that). **Which of the two templates you start from depends on
the Step 2 answers, and you tell the user which one you wrote and why:**

- **The user accepted every default** → write `templates/config.minimal.yml` verbatim as
  `deploy/config.yml` (the filename never changes, only the contents), with the real values
  substituted: `project`, `image`, `kubernetes.context`, `kubernetes.namespace`. Nothing
  else. Say so explicitly,
  e.g. *"you customized nothing, so this is the whole config — five lines; the annotated
  reference with every knob is `templates/config.yml` in the skill, and
  `references/customization.md` is the map"*. Do not pad it with defaults "for
  discoverability": an unset key is a documented default, and the short file is the point
  of the exercise. The four required values do not count as customization; choosing an
  exposure type, a resource override, autoscaling, secrets, a `slug`, or a
  `verify.inference` block does. One thing can still force a fifth line into an otherwise
  all-defaults config: a private registry with no credentials in the namespace needs
  `kubernetes.image_pull_secret` (rule 6 below) — that is a cluster fact, not a
  customization, and it belongs in the minimal file when it applies.
- **Anything was customized** → start from `templates/config.yml`, the annotated
  reference, and keep it annotated (again saved as `deploy/config.yml`). Include the keys
  the user actually chose, plus the comments that explain them; delete the blocks nobody touched rather than writing them
  out at their defaults.

Rules for writing it, either way:

1. **`services:` is optional and holds OVERRIDES ONLY.** No `entry:`, no `depends:` — the
   topology comes from `bento.yaml`. Write a block only for a service that needs one, and
   inside it only the keys that differ from the default. A block keyed to a name that is
   not in `bento.yaml`'s `services[].name` is a load-time **error**, so check the keys
   character by character against the discovered names (that check is now the config's
   whole typo surface).
2. **`expose:` and `ingress:` appear ONLY under the entry service** (`entry_service` in
   `bento.yaml`). On any other service they are rejected at load time (exit 2), because a
   non-entry service reachable from outside the cluster is an unauthenticated pickle
   endpoint. Do not add them "for completeness".
3. **In the annotated file, keep the comments.** They are the documentation for the file
   the user now owns. Adapt them to the real deployment; do not strip them to make the
   output shorter. Comments that only make sense for the template's example can go, and
   later blocks may be terser — but these non-obvious facts must survive: the
   `readiness_timeout_seconds ≥ 6` floor and why; that decorator `resources` are inert in
   OSS so these values are the only ones that bind; that `config_overrides` beats the
   decorator only under the named service; that only the entry service may be exposed
   (pickle → RCE); that quantities are quoted strings; and that the topology, the image
   tag and the build platform are derived, not declared.
4. `project` is a **scalar path** to the project root, resolved relative to `config.yml`
   and required to exist: `..` for the standard `deploy/config.yml` layout.
5. `image` is **one URL with no tag**. Quantities are **quoted strings**. Secrets are
   referenced by name, never inlined.
6. **A private registry needs credentials in the namespace, and that is the one thing the
   four values cannot supply.** Every ECR is private. Check what the namespace already
   has before you decide whether `kubernetes.image_pull_secret` belongs in the file:

   ```bash
   kubectl --context <ctx> -n <ns> get serviceaccount default \
       -o jsonpath='{.imagePullSecrets[*].name}'; echo
   ```

   A non-empty answer means the pods inherit those credentials and the config needs no
   pull-secret key (this is also how a cluster with a node-level credential provider
   behaves — nothing to add). An empty answer with a private `image:` means you must
   either create the secret and set `kubernetes.image_pull_secret`, or have the user
   confirm the nodes carry registry credentials. For ECR, the secret embeds a **12-hour
   token**, so say that out loud: it is fine for a demo and wrong for anything standing —
   the durable fix is a credential provider on the nodes or a refresh job.

   ```bash
   kubectl --context <ctx> -n <ns> create secret docker-registry ecr-creds \
       --docker-server=<account>.dkr.ecr.<region>.amazonaws.com --docker-username=AWS \
       --docker-password="$(aws ecr get-login-password --region <region>)"
   ```

   Preflight reports whichever mechanism it finds (`k8s.image-pull-secret`) and warns when
   it finds none for a private image — that warning is the one from this list you will
   actually see if you skip the check.

Then validate with the bundle's loader — **the only validator there is**:

```bash
grep -n '{{' <project>/deploy/config.yml && echo "ERROR: unreplaced placeholder" || echo "OK"
python3 <project>/deploy/deploy.py --target k8s --check-only --local-only
```

`--check-only --local-only` runs every check that needs no cluster or registry (config,
tooling) and is the same gate a CI PR job uses; drop `--local-only` once the context is
confirmed to also check cluster reachability, RBAC and the namespace. A config error exits
**2** with a path-precise message (e.g. `services.Gateway.expose.node_port: …`) and a hint.
It enforces: every `services:` key names a real service in the bento, slugs DNS-1035 and
unique, resource quantities quoted, `node_port` in range,
`readiness_timeout_seconds ≥ 6` (with the 5 s-fan-out reason), `expose`/`ingress` only on
the entry service, no `BENTOML_RUNNER_MAP` in any `env`, and it **warns** on unknown keys —
treat every warning as a typo to fix, since an ignored key means a knob that silently does
nothing. The whole class of wiring errors it used to check (cycles, unknown `depends`
targets, two entry services, a middle tier with no `depends`) no longer exists: the input
that could be wrong is gone.

Two warnings ARE expected here and must not be "fixed":
`working tree is dirty — the git-SHA image tag will not uniquely identify this build`
(you have not committed `config.yml` yet; it goes away once you do), and, on a project
without a git checkout, the missing-tag notice — both are why Step 5 passes `--image`
explicitly.

Tell the user to **commit `config.yml`**: it is the deployment config and contains no
secrets.

One handoff note for CI: this step pinned the image with `--image <reviewed ref>`, but an
agent-free `deploy.py --target k8s` run builds the bento and tags the image with the bento
version (the project's current short git SHA). So CI's first run re-tags and restarts the
pods once, reporting every Deployment `configured`. That is correct — just not silent. Pass
`--image` (or `--version`) in CI too if you want the exact ref you reviewed here.

## Step 5 — Render the manifests and show them for review

Render into `<project>/k8s/` and show the user what will be applied. Nothing touches the
cluster in this step.

```bash
IMAGE=<full pushed image ref>          # from the containerize handoff, tag included
python3 <project>/deploy/deploy.py --target k8s --render-only k8s --image "$IMAGE"
```

**Always pass `--image` explicitly.** Without it the bundle derives the tag from the bento
version (the project's short git SHA), which renders a tag that is not the image you just
pushed (and hard-fails outside a git checkout). `--render-only` writes the files, prints
the rollout order it would use, and exits without contacting the cluster; with no DIR it
writes to `deploy/rendered/`. It refuses to run while `kubernetes.manifests_dir` is set,
since then nothing is rendered at all.

Rendering needs the topology, so with `--skip-build`-style flow it reads `bento.yaml` out
of the image (docker required) or falls back to the `deploy/.bento-topology.json` cache a
previous run wrote. If it cannot find any of the three, it says which one to produce —
build the bento, or make the image pullable locally, rather than hand-writing anything.

The rendered set, per service, named by slug:

- `k8s/<slug>-deployment.yaml` and `k8s/<slug>-service.yaml` — always, one pair per service.
- `k8s/<slug>-hpa.yaml` — only for services with `autoscaling.enabled: true`.
- `k8s/ingress.yaml` — only when the entry service has `ingress.enabled: true`.

There is **no** rendered `namespace.yaml`: the namespace must already exist (Step 3).

For a single-service bento that is exactly two files, e.g. `k8s/summarization-deployment.yaml`
and `k8s/summarization-service.yaml`.

`templates/*.yaml` in this skill document the **shape** of each rendered file, field by
field, with the `config.yml` key (or the derived source) behind each value. Read them when
reviewing output or when a user asks what a field is for; they are not edited to change a
deployment, and they are not what the renderer reads.

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

Read the `BENTOML_SERVE_DEPENDS` lines against the DAG you reported in Step 1: this is
where you confirm the derived wiring matches the code, and it is the only place it can be
checked before traffic flows. Every non-leaf service must have its own pairs.

`k8s/` is **derived output**. Commit it only if the user wants reviewable diffs or GitOps;
if they do, say plainly that it must be re-rendered after every `config.yml` change and
after every rebuild that changes the topology, and that the source of truth is still
`config.yml` plus the bento. Editing `k8s/` by hand is only correct after deliberately
switching to `kubernetes.manifests_dir` (Step 8).

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
- If the fix involves **renaming** (e.g. setting a `slug` override), `kubectl apply` would
  create a second object under the new name instead of updating the old one — first delete
  the just-created misnamed objects
  (`kubectl --context <ctx> delete -n <ns> deployment/<old> service/<old>`, only the ones
  created moments ago in this run), then set the slug in `config.yml`, re-render (which
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
   user. Record what you used as `verify.inference` in `config.yml` — that is what turns a
   one-off check into a repeatable one, and it is the block whose absence downgrades the
   bundle's verification to `/readyz` only.
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
   separate check, and it is the one that catches the worst failure mode. A dependency the
   pods were not wired for does **not** produce an error: the caller instantiates it
   in-process, so

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
dependency served *something*, not *who called it*. If a leaf is reached only through a
middle service, but the entry service is what actually calls it, the leaf's counter still
moves and answers stay correct. The discriminating evidence is the **client pod IP in the
dependency's access log**: it must be the pod of the service that declares
`bentoml.depends()` on it, not the entry pod. Get the IPs with

```bash
kubectl --context <ctx> get pods -n <ns> -o custom-columns='POD:.metadata.name,IP:.status.podIP' \
  -l app.kubernetes.io/part-of=<bento-name>
```

and check each dependency's log line against the caller you expect (e.g. a leaf reached
only through a middle tier must show the MIDDLE service's pod IP).

   No access-log line and no request counter on the dependency while the entry service
   returned a correct answer **is** the in-process fallback. Since the wiring is derived
   from `bento.yaml`, the fix is **not** in `config.yml`: check that the middle service
   really declares `bentoml.depends()` on the leaf in the source, that `bento.yaml`'s
   `services[].dependencies` shows the edge, and that the rendered
   `BENTOML_SERVE_DEPENDS` on the caller names it (Step 5's grep). A stale
   `deploy/.bento-topology.json` cache used for a rebuilt bento can also produce
   yesterday's wiring — rebuild, re-render and compare. Do not report success.

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

- **Workload shape** (replicas, resources, probes, node placement, exposure, autoscaling):
  `config.yml` → re-render → apply.
- **Runtime behaviour** (timeouts, workers, concurrency limits, logging, CORS, tracing):
  `services.<Name>.config_overrides` in the same file — no image rebuild needed. See
  `references/customization.md`.
- **Topology** (a new service, a new `bentoml.depends()` edge, a renamed service): change
  the **source**, rebuild, re-render. `config.yml` does not change at all — except that a
  `services:` override block keyed to a service you renamed becomes an error, which is the
  point. This is the one kind of change that used to require editing two places.
- **New image**: re-render with the new `$IMAGE` and apply; `config.yml` does not change
  (it holds the repository, never the tag — the tag is the bento version).
- **Outgrowing the config** (sidecars, volumes, PDBs, affinity, `path_prefix` probe
  paths, `terminationGracePeriodSeconds`): render once, commit `k8s/`, set
  `kubernetes.manifests_dir: k8s` and take ownership. Say what is lost as well as gained —
  the config no longer shapes the workload, and the derived guarantees (rollout order,
  `BENTOML_SERVE_DEPENDS`, ClusterIP-only dependencies, HPA-vs-replicas, the image
  reference) become the user's job. `templates/*.yaml` is the checklist of what those
  files must keep.
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

- `templates/config.minimal.yml` — the all-defaults config: `project`, `image`, `kubernetes.context`, `kubernetes.namespace`. What you write when the user customized nothing.
- `templates/config.yml` — the annotated reference for the `bentoml-deploy-config/v4` schema: every knob, at its default, with the reason behind it. The starting point when anything was customized.
- `templates/{deployment,service,hpa,ingress}.yaml` — the documented shape of each rendered object (review reference, and the checklist for `manifests_dir` users). Not read by the renderer.
- `templates/namespace.yaml` — the optional Namespace object; nothing renders one, so apply it yourself if the user wants it in git.

Rendering and config validation live in the sibling `bentoml-deploy-scriptgen` skill's
`templates/deploy/` bundle, copied into the project in Step 4. There is deliberately no
second implementation here.

## References (read on demand)

- `references/customization.md` — the full "what can I customize" map: `config.yml` keys vs. `config_overrides` env knobs vs. what is derived and no longer configurable, plus multi-service gotchas and HPA metric details.
- `references/exposure-options.md` — choosing and configuring port-forward / NodePort / LoadBalancer / Ingress, TLS, inference-friendly timeouts.
- `references/private-registries.md` — imagePullSecrets for GHCR/ECR/Docker Hub/self-hosted; kind/minikube local image loading; ttl.sh for throwaway tests.
- `references/gpu-scheduling.md` — verifying the NVIDIA device plugin, requesting `nvidia.com/gpu`, node selectors and taints.
