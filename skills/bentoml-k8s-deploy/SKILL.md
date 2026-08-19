---
name: bentoml-k8s-deploy
description: Deploy a containerized BentoML service to a vanilla Kubernetes cluster using plain kubectl manifests (no Helm, no operators, no BentoCloud/Yatai). Takes a pushed container image (from the bentoml-containerize skill), discovers the bento's service topology, writes one `config.yml` for the deployment, renders one Deployment + Service per BentoML service (plus optional HPA/Ingress) from it, applies them in dependency order, and verifies the rollout with a real inference request. Use when the user says things like "deploy my BentoML service to Kubernetes", "deploy this bento image to my cluster", "run my bento on k8s", "create k8s manifests for my bento", "split my bento services into separate pods", or "expose my BentoML service in Kubernetes". Also diagnoses a deployment that went wrong — "my BentoML pods are crashing", "ImagePullBackOff", "pod stuck Pending", "readiness probe failing", "rollout stuck", "can't reach my service on Kubernetes", "inference 4xx/5xx".
license: Apache-2.0
compatibility: >-
  Requires kubectl with access to a Kubernetes cluster, Python >= 3.9 with PyYAML (the bundled renderer runs locally), and an image registry the cluster can pull from; AWS CLI v2 for ECR.
---


# Deploy a BentoML service to vanilla Kubernetes

Take a pushed image ref, discover the bento's services, write one `deploy/config.yml`,
render plain manifests from it, apply them, verify with a real inference request.

**`config.yml` is the only file the user edits.** It holds the image URL, the
cluster/namespace, and optional per-service overrides (replicas, resources, probes, env,
autoscaling, exposure). Manifests are rendered output: "change the deployment" means edit
the config, re-render, re-apply — never edit the rendered YAML.

**The config does not record the topology.** The service list, the entry service and the
dependency DAG come from the bento's own `bento.yaml`, re-read on every run. There is no
`entry:` key and no `depends:` key, so the config can never disagree with the code. A
config that customizes nothing is five lines (`templates/config.minimal.yml`).

**One Deployment + Service per BentoML service**, so each can be sized and scaled
independently. A single-service bento is the same path with one pair — do not special-case it.

Derived, never asked and never written down:

| Derived | From |
|---|---|
| Service list, entry service, DAG | `bento.yaml` (`services[].name`, `entry_service`, `services[].dependencies[].service`) |
| Rollout order | topological sort of the DAG — deepest tier first, alphabetical within a tier |
| `BENTOML_SERVE_DEPENDS` | the DAG + each dependency's slug + `kubernetes.namespace` |
| Slugs, labels, selectors | each service's name |
| ClusterIP on every non-entry Service | `entry_service` (an RCE invariant, see safety rules) |
| Image tag | the bento version |
| Build platform | builder arch vs. node arch; a cross-build gets `--opt platform=linux/<arch>` automatically |
| Registry auth | the `image` URL host (`*.dkr.ecr.<region>.amazonaws.com` ⇒ ECR login + describe-or-create) |
| Omitting `spec.replicas` | `autoscaling.enabled: true` (an HPA and a replica count fight on every apply) |

**Rendering and validation are not implemented here.** They live in the deploy bundle from
the sibling `bentoml-deploy-scriptgen` skill, copied into the project in Step 4 and driven
with `--render-only` / `--check-only` — so what you review is what CI applies later.

Always true for `bentoml containerize` images:

- Port **3000** in every pod; **`/livez`** (liveness), **`/readyz`** (readiness),
  `/metrics`, `/docs.json`.
- Use **`args:`**, never `command:` — `command:` bypasses the entrypoint's venv activation
  and the container dies immediately.
- `args: ["start-http-server", "--service-name", "<ServiceName>"]` serves exactly one
  service and spawns no sibling processes.
- **Never set `BENTOML_RUNNER_MAP` / `BENTOML_SERVE_RUNNER_MAP`.** The serving process
  overwrites it, and any dependency it cannot resolve is then instantiated **in-process** —
  the pod loads every model and still reports healthy. Wire with `BENTOML_SERVE_DEPENDS`
  only, which the renderer derives.
- `@bentoml.service(resources={"cpu"/"memory"})` is **inert in open-source BentoML**, so the
  config's resource values are the only thing constraining a pod.
- Models are usually baked into the image; runtime downloads (gated HF models) need env vars
  such as `HF_TOKEN` from a Secret.

Safety rules: never apply before the user confirms the context **and** namespace; never
`kubectl delete` anything this skill did not create in this session; never write secret
values into any file (create Secrets with `--from-literal`, reference them by name); and
never give a **non-entry** service a Service type other than `ClusterIP` — inter-service
traffic is `application/vnd.bentoml+pickle`, i.e. unauthenticated pickle deserialization, so
exposing it is remote code execution (the renderer rejects `expose:`/`ingress:` anywhere but
the entry service).

When any step below fails, go to `references/troubleshooting.md` — it is indexed by symptom.

## Step 0 — Preflight

```bash
kubectl version --client
python3 -c 'import yaml; print("pyyaml", yaml.__version__)'   # config.yml is YAML; stdlib has no parser
kubectl config get-contexts                                   # note the current one (*)
```

**Ask which context to use — never assume the current one**, even if it is the only one.
Then pass `--context <ctx>` on every command (never `kubectl config use-context`) and record
it as `kubernetes.context`.

```bash
kubectl --context <ctx> get nodes -L kubernetes.io/arch
kubectl --context <ctx> auth can-i create deployment -n <ns>   # once the namespace is known (Step 3)
```

The `ARCH` column matters for a **pre-built** image only: there is no platform knob, and a
building run cross-builds automatically. But an image built on Apple Silicon for amd64 nodes
crash-loops with `exec format error` — if the arches differ, rebuild rather than deploy.

The user also needs an image registry the cluster can pull from (or a local cluster with the
image loaded onto the nodes — `references/private-registries.md`).

## Step 1 — Discover the topology

Do this before asking about sizing. It produces a **report** (what the bento contains) and
**sizing suggestions** — and nothing that goes into a file: `deploy.py` rediscovers the
topology on every run from the built bento, from the image under `--skip-build`, or from the
`deploy/.bento-topology.json` cache. Tell the user that, so they know nothing can drift.

Primary source, `bento.yaml` inside the image (no user code imported):

```bash
docker run --rm --entrypoint cat <image> /home/bentoml/bento/bento.yaml
# a custom base image may move it:
docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' <image> | grep '^BENTO_PATH='
```

`docker pull` first if the image is remote; `podman`/`nerdctl` work the same way. If the
bento is in the local store, `bentoml get <tag> -o json` has the same content and needs no
docker.

| Field | Use |
|---|---|
| `name` | bento name → `app.kubernetes.io/part-of`, Secret name prefix, how to `kubectl get -l` the whole bento |
| `entry_service` | receives external traffic; the only service that may be exposed, that verify targets, and that may carry `expose`/`ingress` |
| `services[].name` | the service set — the **only valid keys** under `services:`, verbatim |
| `services[].dependencies[].service` | DAG edges (values are BentoML service names). Report them; write them nowhere |
| `services[].config` | declared `resources`/`workers`/`traffic` → Step 2 suggestions |

`services[]` is not in dependency order. Report the derived order so the user can
sanity-check the DAG:

```
bento gateway: 4 services (entry_service: Gateway)
  Gateway   (entry) -> Enricher, Sentiment      [fan-out]
  Enricher          -> Tokenizer                [middle tier]
  Sentiment         -> (leaf)
  Tokenizer         -> (leaf, 2 hops from the entry)
Rollout order (derived at deploy time): Sentiment, Tokenizer, Enricher, Gateway
```

**If the image is not built yet**, read `service.py` — but `bento.yaml` is authoritative, so
build and re-check before shipping anything derived this way. Each `@bentoml.service(...)`
class is a service (decorator kwargs = declared config), each `x = bentoml.depends(Other)` is
an edge, and the build target (`bentoml build service:Gateway`, or `service:` in
`pyproject.toml`/`bentofile.yaml`) is the entry service — otherwise the one nothing depends
on; if several qualify, ask, and record `build_target`. Two traps:

- **A service's name is the decorator's `name=` kwarg when present, else the class name**
  (BentoML resolves `name or inner.__name__`), so `@bentoml.service(name="sentiment-v2")
  class Sentiment` is `sentiment-v2`. Grep for `name=` in every decorator: `--service-name`
  fails **loudly** if you get it wrong, but a dependency key fails **silently** — an
  unmatched key instantiates the dependency in-process.
- **`path_prefix="/x"`** moves that service's `/livez`, `/readyz` and `/metrics` under `/x`,
  and the renderer has no knob for it. Flag it and treat that service as a `manifests_dir`
  case, or drop the prefix — never leave the probes at the root.

### Slugs are DNS-1035

Each service gets a slug — its service name snake_cased with `_` → `-`: `TextPipeline` →
`text-pipeline`, `TTSService` → `tts-service`. The renderer derives them and `--render-only`
prints them, so report the expected names with the topology rather than computing them by
hand. Kubernetes Service names are RFC 1035 — stricter than most:
`^[a-z]([-a-z0-9]*[a-z0-9])?$`,
must **start with a letter**, max 63 chars.

- The renderer validates this and exits 2 naming `services.<Name>.slug`. A name starting
  with a digit needs an explicit `slug`.
- Slugs must be unique; on a collision ask which to rename and write the override.
- Never derive a slug from the image repo basename (a ttl.sh UUID starts with a digit).
- **`--dry-run=client` does not catch DNS-1035 violations for Services** — a bad name
  passes dry-run, then the real apply rejects the Service while the Deployment succeeds,
  leaving a half-applied state (Step 6).

## Step 2 — One round of questions

Ask everything at once, and lead with the fact that **every question has a working
default**: "all defaults" gives a five-line config. After this, changes are config edits.

Bento-wide:

| Parameter | Config key | Default | Notes |
|---|---|---|---|
| Image repository | `image` | required | **One URL, no tag** — the tag is the bento version. Auth derived from the host. For kind/minikube, load the image on the nodes and set `image: ""`. |
| Namespace | `kubernetes.namespace` | `default` | Offer a dedicated one. All services share it — dependency DNS names are rendered with it. |
| Project root | `project` | `..` | Scalar path relative to `config.yml`. |
| Build target | `build_target` | omitted | Only when plain `bentoml build` cannot resolve which service to build. |
| Secrets | `services.<Name>.env_from_secrets` | none | Created in Step 3; only the Secret **name** reaches the config. |
| Pull secret | `kubernetes.image_pull_secret` | detect/ask | See rule 6 in Step 4. |
| Exposure (entry only) | `services.<Entry>.expose` / `.ingress` | port-forward (ClusterIP) | Read `references/exposure-options.md` and check what the cluster supports (`kubectl get ingressclass`, LB availability). Non-entry services are always ClusterIP — do not offer a choice. |
| Verification call | `verify.inference` | omitted | With it, verification proves the model answers *and* that dependencies were called in their own pods. Without it, `/readyz` only and the dependency proof is skipped — worth two minutes on a multi-service bento. |

Per service, all **overrides** — omit the key or the whole block for the default:

| Parameter | Config key | Default | Notes |
|---|---|---|---|
| Replicas | `replicas` | `1` | Scale each service independently; the bottleneck is rarely the entry. |
| CPU req/limit | `resources.requests.cpu` / `.limits.cpu` | `"500m"` / `"2"` | If the decorator declares `resources={"cpu": "1"}`, suggest mirroring it here **and say why**: the declaration does nothing in OSS, so this is the only place it binds. |
| Memory req/limit | `resources.requests.memory` / `.limits.memory` | `"1Gi"` / `"4Gi"` | Size to the model; OOMKilled usually means the limit is below its real footprint. |
| GPU | `resources.limits["nvidia.com/gpu"]` | none | `resources.gpu` is the one decorator key that does work in OSS (it sets `CUDA_VISIBLE_DEVICES`), so a service declaring it must request the device here. See `references/gpu-scheduling.md`; pair with `node_selector`/`tolerations`. |
| Workers | `config_overrides.workers` | decorator, else `1` | Server-side, rendered into `BENTOML_CONFIG_OVERRIDES`. N workers want ~N cores. |
| Timeout | `config_overrides.traffic.timeout` | decorator, else `60` | Server-side. Raising a **dependency's** timeout needs the caller's block too — `references/customization.md`. |
| Autoscaling | `autoscaling.*` | `enabled: false` | Per service; ask min/max and target (CPU % or in-flight requests). The renderer then omits `spec.replicas`. |
| Plain env | `env` | none | Non-sensitive only. |
| Placement | `node_selector`, `tolerations` | none | GPU pools, arch pinning, tainted groups. |
| Object name | `slug` | derived | Only for a collision or a legacy name. |

Note the asymmetry: a decorator value still governs the server (`workers`,
`traffic.timeout`), but decorator `resources.cpu`/`memory` govern **nothing** — those are
the two you actively suggest mirroring into the config.

**Probe-timeout floor.** `probes.readiness_timeout_seconds` defaults to `6`, and 6 is a
**floor**. A `/readyz` that fans out to a `bentoml.depends()` dependency gives it a
hard-coded **5-second** budget, so `/readyz` can legitimately take just over 5 s. At `3`
the kubelet hangs up first and a dependency answering in 4 s leaves the caller
**permanently unready**, with nothing in its own logs. The 5 s is a framework constant
(`runner_probe.timeout` is inert on BentoML 1.2+). Raise it for a deep chain, never lower
it, and keep leaf services on the same value so adding a dependency later cannot break
them. Liveness stays at `timeoutSeconds: 3` (not configurable) — `/livez` is pod-local.

**Quote every quantity.** `cpu: "1"`, `cpu: "500m"`, `memory: "4Gi"`,
`nvidia.com/gpu: "1"`. Unquoted `cpu: 1` is a YAML integer while the API server stores the
string `"1"`, so the merge patch differs on **every** apply: `kubectl apply` reports
`configured` forever instead of `unchanged`, and GitOps sees phantom drift. The renderer
rejects unquoted quantities so the error names the user's own line.

Anything else (concurrency limits, access logging, CORS, tracing) goes in
`config_overrides`, not a Kubernetes field — `references/customization.md` splits the whole
surface into what belongs in `config.yml`, what belongs in `config_overrides`, what only
looks tunable, and what was BentoCloud-only. Two rules:

- The renderer nests what you write under the **named service**. A global
  `{"services":{"<key>":...}}` merges *underneath* each service's decorator config and
  loses to it. Write bare keys (`workers`, `traffic`, `logging`).
- `<ServiceName>` is the BentoML service name verbatim, never the slug — which is why
  `services:` blocks are keyed by service name.

## Step 3 — Namespace and Secrets (imperative, before the config)

Nothing downstream creates a namespace: the bundle refuses to run against a missing one and
never renders a Namespace object. (Apply `templates/namespace.yaml` yourself if the user
wants it in git.)

```bash
kubectl --context <ctx> get namespace <ns> || kubectl --context <ctx> create namespace <ns>
```

One application Secret per bento, shared by the services that need it (per-service Secrets
only if values differ). Ask for the values or confirm reading them from the shell env; never
echo them, never put them in a file:

```bash
kubectl --context <ctx> create secret generic <bento-slug>-env -n <ns> \
  --from-literal=HF_TOKEN="$HF_TOKEN"
```

→ `services.<Name>.env_from_secrets: [<bento-slug>-env]`.

Pull secret for a private registry, one per namespace (all services run the same image):

```bash
kubectl --context <ctx> create secret docker-registry <bento-slug>-regcred -n <ns> \
  --docker-server=<registry> --docker-username=<user> --docker-password="$REGISTRY_TOKEN"
```

→ `kubernetes.image_pull_secret: <bento-slug>-regcred`. Per-registry details:
`references/private-registries.md`.

## Step 4 — Install the deploy bundle, then write `config.yml`

Rendering and validation come from the sibling `bentoml-deploy-scriptgen` skill's bundle.
**Do not write a second renderer.** The skills install together, so the sibling is normally
the directory next to this one; if it is genuinely missing, say so and stop.

```bash
SCRIPTGEN=<path-to-bentoml-deploy-scriptgen-skill>
mkdir -p <project>/deploy
cp -R "$SCRIPTGEN/templates/deploy/." <project>/deploy/
find <project>/deploy -name __pycache__ -type d -prune -exec rm -rf {} +   # mandatory
find <project>/deploy -name '*.pyc' -delete
```

Then write `<project>/deploy/config.yml` (next to `deploy.py`, where it is found by
default). Which template you start from depends on the Step 2 answers, and you tell the user
which one you wrote:

- **Everything defaulted** → `templates/config.minimal.yml` with the four real values, and
  nothing else. Tell the user that is the whole config and point at `templates/config.yml`
  for the knobs. Do not pad it with defaults for discoverability. An exposure type, a
  resource override, autoscaling, secrets, a `slug` or `verify.inference` counts as
  customization; the four required values do not. A private registry with no credentials in
  the namespace can force a fifth line (`kubernetes.image_pull_secret`) — a cluster fact,
  not a customization.
- **Anything customized** → start from `templates/config.yml`, keep it annotated, include
  only the keys the user chose, delete the untouched blocks.

Rules either way:

1. **`services:` is optional and holds overrides only.** No `entry:`, no `depends:`. A key
   that names no service in `bento.yaml` is a load-time error — check them character by
   character; that is the config's whole typo surface.
2. **`expose:`/`ingress:` only under the entry service.** Rejected elsewhere at load time
   (exit 2) because a non-entry service reachable from outside is an unauthenticated pickle
   endpoint. Do not add them "for completeness".
3. **Keep the comments in the annotated file** — they document the file the user now owns.
   Adapt them to the real deployment. These must survive: the `readiness_timeout_seconds ≥ 6`
   floor and why; that decorator `resources` are inert in OSS; that `config_overrides` beats
   the decorator only under the named service; that only the entry service may be exposed
   (pickle → RCE); that quantities are quoted; that the topology, tag and platform are
   derived.
4. `project` is a scalar path relative to `config.yml` and must exist.
5. `image` is one URL with no tag. Quantities are quoted. Secrets by name only.
6. **A private registry needs credentials in the namespace — the one thing the four values
   cannot supply.** Every ECR is private. Check what is already there:

   ```bash
   kubectl --context <ctx> -n <ns> get serviceaccount default \
       -o jsonpath='{.imagePullSecrets[*].name}'; echo
   ```

   Non-empty → the pods inherit those credentials and the config needs no pull-secret key
   (same for a node-level credential provider). Empty with a private `image:` → create the
   secret and set `kubernetes.image_pull_secret`, or have the user confirm the nodes carry
   credentials. For ECR the secret embeds a **12-hour token**: fine for a demo, wrong for
   anything standing — the durable fix is a node credential provider or a refresh job.

   ```bash
   kubectl --context <ctx> -n <ns> create secret docker-registry ecr-creds \
       --docker-server=<account>.dkr.ecr.<region>.amazonaws.com --docker-username=AWS \
       --docker-password="$(aws ecr get-login-password --region <region>)"
   ```

   Preflight reports whichever mechanism it finds (`k8s.image-pull-secret`) and warns when a
   private image has none.

Validate with the bundle's loader — the only validator there is:

```bash
grep -n '{{' <project>/deploy/config.yml && echo "ERROR: unreplaced placeholder" || echo "OK"
python3 <project>/deploy/deploy.py --target k8s --check-only --local-only
```

`--local-only` skips everything needing a cluster or registry (the same gate a CI PR job
uses); drop it to also check reachability, RBAC and the namespace. A config error exits **2**
with a path-precise message. It enforces: `services:` keys name real services, slugs
DNS-1035 and unique, quantities quoted, `node_port` in range,
`readiness_timeout_seconds ≥ 6`, `expose`/`ingress` only on the entry service, no
`BENTOML_RUNNER_MAP` in any `env` — and it **warns** on unknown keys, which always mean a
typo, i.e. a knob doing nothing.

Two warnings are expected here: the dirty-working-tree note (you have not committed
`config.yml` yet) and, without a git checkout, the missing-tag notice — both are why Step 5
passes `--image` explicitly. Tell the user to **commit `config.yml`**; it holds no secrets.

CI note: an agent-free `deploy.py --target k8s` run builds the bento and tags with the bento
version, so CI's first run re-tags and restarts the pods once, reporting every Deployment
`configured`. Correct, just not silent — pass `--image`/`--version` in CI too to pin the ref
you reviewed here.

## Step 5 — Render and review

Nothing touches the cluster here.

```bash
IMAGE=<full pushed image ref>          # from the containerize handoff, tag included
python3 <project>/deploy/deploy.py --target k8s --render-only k8s --image "$IMAGE"
```

**Always pass `--image`.** Without it the tag comes from the bento version (the project's
short git SHA), which is not the image you just pushed — and it hard-fails outside a git
checkout. `--render-only` writes the files, prints the rollout order and exits without
contacting the cluster (default DIR `deploy/rendered/`); it refuses to run while
`kubernetes.manifests_dir` is set. It needs the topology, so it reads `bento.yaml` from the
image (docker required) or the `.bento-topology.json` cache, and names which one to produce
if neither is there — build the bento rather than hand-writing anything.

Rendered set, per service, named by slug: `<slug>-deployment.yaml` and
`<slug>-service.yaml` always; `<slug>-hpa.yaml` when `autoscaling.enabled`; one
`ingress.yaml` when the entry service enables it. There is **no** rendered `namespace.yaml`.

`templates/*.yaml` in this skill document the **shape** of each rendered file with the
config key behind each value — read them when reviewing output, not to change a deployment.

Sanity-check what landed:

```bash
grep -n '{{' k8s/*.yaml && echo "ERROR: unreplaced placeholders" || echo "OK"
grep -rn '^[^#]*command:' k8s/ && echo "ERROR: never override command:"
grep -rn '^[^#]*RUNNER_MAP' k8s/ && echo "ERROR: never set the runner map"
grep -hE '^\s+(cpu|memory|nvidia\.com/gpu):' k8s/*-deployment.yaml \
  | grep -v '"' && echo "ERROR: unquoted quantity" || echo "OK: all quantities quoted"
grep -H 'type:' k8s/*-service.yaml            # only the entry one may be non-ClusterIP
grep -A1 -n 'BENTOML_SERVE_DEPENDS' k8s/*-deployment.yaml
kubectl --context <ctx> apply -n <ns> --dry-run=client -f k8s/
```

Read the `BENTOML_SERVE_DEPENDS` lines against the DAG from Step 1 — this is the only place
the derived wiring can be checked before traffic flows, and every non-leaf service must have
its own pairs.

`k8s/` is derived output. Commit it only for reviewable diffs or GitOps, and say that it must
be re-rendered after every config change and every rebuild that changes the topology.
Hand-editing it is only correct after deliberately switching to `manifests_dir` (Step 8).

## Step 6 — Confirm, then apply in the derived order

Show the rendered manifests plus one summary line —
`context=<ctx>  namespace=<ns>  image=<image>  services=<slug1>,<slug2>(entry)` — and wait
for explicit confirmation.

Apply in the derived order (dependencies first, entry last), waiting for each rollout. An
entry pod started before its dependencies exist sits un-ready, because its `/readyz` fans
out to them:

```bash
ROLLOUT=<kubernetes.rollout_timeout_seconds>   # 900 by default; NOT the startup budget
for SLUG in sentiment text-pipeline; do        # order from --render-only, deepest first
  kubectl --context <ctx> apply -n <ns> -f k8s/$SLUG-deployment.yaml -f k8s/$SLUG-service.yaml
  [ -f k8s/$SLUG-hpa.yaml ] && kubectl --context <ctx> apply -n <ns> -f k8s/$SLUG-hpa.yaml
  kubectl --context <ctx> rollout status deployment/$SLUG -n <ns> --timeout=${ROLLOUT}s
done
[ -f k8s/ingress.yaml ] && kubectl --context <ctx> apply -n <ns> -f k8s/ingress.yaml
```

A bulk `apply -f k8s/` also converges (the 10-minute startupProbe covers the wait) and is
what the generated script does; ordered apply just gives clearer failures.

**If apply partially fails** — fix `config.yml`, re-render, re-apply. Resources this run
just created are yours to delete and re-apply; the never-delete rule protects pre-existing
ones. For a **rename** (setting a `slug`), `apply` would create a second object, so delete
the just-created misnamed pair first, then re-render (which updates the callers' dependency
URLs) and apply.

## Step 7 — Verify

**1. Every rollout**, same order. `kubernetes.rollout_timeout_seconds` (900) is deliberately
longer than the startup budget (`probes.startup_failure_threshold` × 10 s = 600 s): equal
values make a pod that uses its whole budget lose the race and be reported failed at the
moment it succeeded.

```bash
kubectl --context <ctx> get pods -n <ns> -l app.kubernetes.io/part-of=<bento-name>
```

A stalled rollout goes to `references/troubleshooting.md`. A dependency that is up but
unreachable makes the *caller* un-ready — check the dependency's endpoints and that the
rendered `BENTOML_SERVE_DEPENDS` hostname matches its Service name exactly.

**2. One real inference request, against the ENTRY service only.** Do not call a dependency
with JSON — it speaks `application/vnd.bentoml+pickle` and will reject you. Exercising the
entry endpoint is what proves the wiring.

Derive the endpoint from the entry service's API (`def foo(self, text: str)` is
`POST /foo` with `{"text": ...}`): `bento.yaml`'s `schema.routes`, or `/docs.json` while the
tunnel is up, or ask. Record it as `verify.inference` — that is what makes the check
repeatable, and its absence downgrades the bundle to `/readyz` only.

Forward to `kubernetes.local_port` (e.g. 3130), **not local 3000**: dev servers squat there,
and a port-forward that fails to bind sends your curls to whatever is listening instead —
convincing, fake results. Hence the tunnel liveness check. Run the whole block in **ONE shell
invocation** (`$PF_PID` does not survive separate calls):

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
[ -n "$OK" ] || { echo "not ready after 30s:"; cat "$PF_ERR"; kill $PF_PID; exit 1; }
echo READY
curl -s -X POST http://127.0.0.1:$PORT/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text": "Kubernetes is an open-source container orchestration system."}'
kill $PF_PID; rm -f "$PF_ERR"
```

**Judge the response by its content, not the status code.** A 200 proves nothing. If the
output looks unrelated to the API, suspect a local port squatter.

**3. Multi-service bentos: prove the dependency ran in its OWN pod.** Wiring the pods missed
produces no error: the caller instantiates the dependency in-process, `/readyz` still returns
200 (the fan-out only probes remote proxies) and the answer is completely correct — same code,
wrong pod, every model in the caller. So the evidence is only on the dependency's side. Run
this right after the inference call:

```bash
kubectl --context <ctx> logs -n <ns> -l app.kubernetes.io/name=<dep-slug> --tail=20 --since=2m
# bento images ship neither curl nor wget, so scrape /metrics with the interpreter:
kubectl --context <ctx> exec -n <ns> deploy/<dep-slug> -- python3 -c \
  "import urllib.request as u; print(''.join(l for l in u.urlopen('http://localhost:3000/metrics').read().decode().splitlines(True) if l.startswith('bentoml_service_request_total') and 'livez' not in l and 'readyz' not in l))"
```

A non-zero exit or empty output is **not** proof of the fallback — separate "served nothing"
from "the command failed" first.

**Deeper than two tiers, a moving counter is not enough**: it shows the dependency served
something, not who called it. If the entry service wrongly calls a leaf that should be reached
only through a middle tier, the counter still moves and answers stay correct. The
discriminating evidence is the **client pod IP** in the dependency's access log — it must be
the pod that declares `bentoml.depends()` on it:

```bash
kubectl --context <ctx> get pods -n <ns> -l app.kubernetes.io/part-of=<bento-name> \
  -o custom-columns='POD:.metadata.name,IP:.status.podIP'
```

No access-log line and no counter movement while the entry service answered correctly **is**
the in-process fallback. The wiring is derived, so the fix is not in `config.yml`: check the
source really declares the `bentoml.depends()`, that `bento.yaml` shows the edge, and that
the rendered `BENTOML_SERVE_DEPENDS` names it (Step 5's grep). A stale
`.bento-topology.json` can also serve yesterday's wiring — rebuild, re-render, compare. Do
not report success.

## Step 8 — How to reach it, and how to change it

Access applies to the **entry service** only (details in `references/exposure-options.md`):

- **port-forward**:
  `kubectl --context <ctx> port-forward svc/<entry-slug> -n <ns> <local_port>:3000`
- **NodePort**: `http://<node-ip>:<node-port>` (`kubectl get svc`, `kubectl get nodes -o wide`)
- **LoadBalancer**: external address from `kubectl get svc <entry-slug> -n <ns> -w`
- **Ingress**: `http(s)://<ingress.host>/` once DNS points at the controller

Also mention Swagger UI at `/`, per-pod metrics at `/metrics`, and that dependency Services
are cluster-internal by design.

The change loop is one loop:

```bash
# edit deploy/config.yml, then:
python3 deploy/deploy.py --target k8s --render-only k8s --image "$IMAGE"
kubectl --context <ctx> apply -n <ns> -f k8s/
```

Or hand it to the bundle: `deploy.py --target k8s --skip-build --image "$IMAGE"` applies in
rollout order, waits and verifies.

| Change | What to do |
|---|---|
| Workload shape (replicas, resources, probes, placement, exposure, autoscaling) | `config.yml` → re-render → apply |
| Runtime behaviour (timeouts, workers, concurrency, logging, CORS, tracing) | `services.<Name>.config_overrides` — no rebuild |
| Topology (new service, new `depends()`, rename) | change the **source**, rebuild, re-render. `config.yml` does not change — except that an override block keyed to a renamed service becomes an error, which is the point |
| New image | re-render with the new `$IMAGE` and apply; the config holds the repository, never the tag |
| Production / CI-CD | nothing to migrate: commit `deploy/` and run it from CI |

**Outgrowing the config** (sidecars, volumes, PDBs, affinity, `path_prefix` probe paths,
`terminationGracePeriodSeconds`): render once, commit `k8s/`, set
`kubernetes.manifests_dir: k8s`. Say what is lost too — the config no longer shapes the
workload, and rollout order, `BENTOML_SERVE_DEPENDS`, ClusterIP-only dependencies,
HPA-vs-replicas and the image reference all become the user's job. `templates/*.yaml` is the
checklist of what those files must keep.

## Scope

Autoscaling is the plain HorizontalPodAutoscaler rendered from `autoscaling.*` (CPU out of
the box; in-flight-request scaling needs prometheus-adapter or KEDA). Scale-to-zero and
canary/blue-green were BentoCloud features and are **not** in scope — say so and stop rather
than improvising.

## Files in this skill

- `templates/config.minimal.yml` — the all-defaults config (four values).
- `templates/config.yml` — the annotated `bentoml-deploy-config/v4` reference: every knob at
  its default, with the reason behind it.
- `templates/{deployment,service,hpa,ingress}.yaml` — the documented shape of each rendered
  object (review reference, and the checklist for `manifests_dir` users). Not read by the
  renderer.
- `templates/namespace.yaml` — optional; nothing renders a Namespace.

## References (read on demand)

- `references/troubleshooting.md` — symptom-indexed runbook: ImagePullBackOff,
  CrashLoopBackOff, OOM, Pending, probe failures, unreachable Services, inference 4xx/5xx,
  and the silent in-process dependency fallback. **Read it whenever a step above fails**, or
  when the user arrives with a broken deployment instead of a new one.
- `references/customization.md` — what belongs in `config.yml` vs. `config_overrides` vs.
  what is derived, plus multi-service gotchas and HPA metric details.
- `references/exposure-options.md` — port-forward / NodePort / LoadBalancer / Ingress, TLS,
  inference-friendly timeouts.
- `references/private-registries.md` — imagePullSecrets for GHCR/ECR/Docker Hub/self-hosted;
  kind/minikube image loading; ttl.sh for throwaway tests.
- `references/gpu-scheduling.md` — NVIDIA device plugin, requesting `nvidia.com/gpu`, node
  selectors and taints.
