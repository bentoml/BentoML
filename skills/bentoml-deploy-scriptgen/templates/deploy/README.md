# deploy/ — production deploy bundle for {{SERVICE_NAME}}

Standalone, agent-free deploy scripts from the `bentoml-deploy-scriptgen` skill. Needs
Python >= 3.9 and PyYAML. `bentoml`, `docker`, `kubectl` and `aws` run via subprocess,
validated by preflight before anything is mutated.

**`config.yml` is the only file you edit** — never `deploy.py` or `_internal/`.
Manifests are rendered on every run from `config.yml` plus the bento's own service
topology and piped straight to `kubectl apply`; no hand-owned YAML to keep in sync. If
`config.yml` cannot express what you need, regenerate the bundle with the skill, or take
over the YAML with `kubernetes.manifests_dir` (below).

**`config.yml` holds overrides only.** Read from the bento's `bento.yaml` at run time
and not settable here: the service list, which service is the entry service, the
dependency DAG, and everything derived from them (rollout order,
`BENTOML_SERVE_DEPENDS`, slugs, labels, ClusterIP on every non-entry Service). Also
derived: the image tag (the bento version), the registry type (from the `image:` host),
the build platform (detected).

Four values are enough to deploy:

```yaml
project: ..
image: 123456789012.dkr.ecr.us-west-1.amazonaws.com/my-bento
kubernetes:
  context: my-cluster
  namespace: default
```

That is `config.minimal.yml`, shipped beside the annotated `config.yml`; both are valid
`--config` input. Every service gets the defaults from the table below, the entry Service
is ClusterIP verified through a port-forward, and verification is `/readyz`-only. Add
keys when you need a NodePort, an Ingress, autoscaling, per-service resources or an
inference smoke test.

One prerequisite no config can supply: **the cluster must already be able to pull the
image.** A private registry (every ECR is one) needs credentials in the namespace:
`kubernetes.image_pull_secret` (a fifth value), an `imagePullSecrets` on the namespace's
`default` serviceaccount, or a node-level credential provider. Preflight reads that
serviceaccount and reports which mechanism it found; for a private image with none, it
warns with the exact `kubectl create secret docker-registry` line.

PyYAML is the one third-party requirement (the standard library has no YAML parser);
`bentoml` depends on it. A deploy-only environment needs `pip install pyyaml`
(`common.pyyaml` in preflight).

## Usage

```bash
# Preflight only — safe anywhere, changes nothing:
python3 deploy/deploy.py --target k8s --check-only

# Same, without cluster/registry credentials (CI PR gate):
python3 deploy/deploy.py --target k8s --check-only --local-only

# Show what would be applied (writes deploy/rendered/, applies nothing):
python3 deploy/deploy.py --target k8s --render-only

# Full pipeline: build, containerize, push, render, apply, verify.
# The image tag defaults to the project's short git SHA:
python3 deploy/deploy.py --target k8s

# Re-running at the same git SHA is idempotent: the built bento is reused. A forced
# REBUILD at the same tag (bentoml delete + re-run) reassigns the tag to a new digest,
# leaves the previous image untagged, and nodes that cached the old digest keep it
# under imagePullPolicy IfNotPresent — prefer a new commit (new tag).

# Deploy an already-pushed image (no build), e.g. rollback. Under --skip-build the
# topology comes from the image:
#   docker run --rm --entrypoint cat <image> $BENTO_PATH/bento.yaml
python3 deploy/deploy.py --target k8s --skip-build --image {{IMAGE_URL}}:<tag>

# EC2: the same pipeline onto the instances in ec2.hosts
# (export the values for every name in ec2.env_names first):
python3 deploy/deploy.py --target ec2 --check-only   # probes every host over SSH
python3 deploy/deploy.py --target ec2
```

### Flags

| Flag | Meaning |
|---|---|
| `--target {k8s,ec2}` | Required. A target not generated into this bundle exits 2. |
| `--check-only` | Run every applicable preflight check in one aggregated pass and exit; nothing is changed. |
| `--local-only` | With `--check-only`: only checks needing no cluster or registry (config validity, rendering, tool presence). Skipped ones are listed in the summary's `skipped_checks`. |
| `--render-only [DIR]` | k8s only. Render the manifests into `DIR` (default `deploy/rendered/`) and exit 0 **without touching the cluster** — nothing built, applied or contacted (see below). Not combinable with `--check-only`; run them one after the other. |
| `--skip-build` | Skip build/containerize/push and their preflights; deploy an existing image. Preflight verifies the image exists in the registry (ECR `describe-images`, else `docker manifest inspect`) so a typo cannot burn the rollout timeout. |
| `--image REF` | Full image reference — repository **and** tag. Bypasses build and push, and overrides `image:`. |
| `--version TAG` | The **bento version**, also the image tag (`bentoml build --version TAG`). Default `git rev-parse --short HEAD` of the project, so bento version == git SHA == image tag; a non-git project fails preflight with instructions. |
| `--no-verify` | Skip the post-deploy `/readyz` + inference verification. |
| `--config PATH` | Alternate config file (default: `config.yml` next to `deploy.py`). |
| `--output-json PATH` | Also write the machine-readable summary to a file. |

Environment overrides (lower precedence than flags): `BENTOML_DEPLOY_IMAGE`,
`BENTOML_DEPLOY_VERSION`, `BENTOML_DEPLOY_K8S_CONTEXT`, `BENTOML_DEPLOY_K8S_NAMESPACE`,
`BENTOML_DEPLOY_EC2_HOSTS` (comma-separated, overrides `ec2.hosts`).

Secrets are **never** read from the config or written to disk. AWS/ECR credentials come
from the ambient AWS environment (`AWS_PROFILE`,
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, or SSO); registry logins from `docker
login`; cluster access from your kubeconfig.

### Exit codes and the JSON summary

`0` ok · `1` generic · `2` config error · `3` preflight failure ·
`4` build/containerize · `5` push · `6` deploy · `7` verify.

The **last line of stdout is always a JSON summary** (progress logging goes to stderr):

```json
{"schema": "bentoml-deploy-summary/v1", "ok": true, "exit_code": 0, "target": "k8s", "action": "deploy", "image": "...", "version": "...", "stages": [...], "skipped_checks": [...], "error": null, ...}
```

`action` is `deploy`, `check-only` or `render-only`. `stages` carries the pipeline steps
(`preflight`, `build`, `containerize`, `push`, `deploy`, `verify`, or the single
`render` stage under `--render-only`) plus one entry per unit of work:

- `k8s.apply` — the `kubectl apply`, with a `detail` naming where the YAML came from:
  `rendered N object(s) from config.yml`, or `applied N file(s) from <dir> VERBATIM
  (kubernetes.manifests_dir) — the image refs in those files are what runs; this run's
  image (…) was NOT substituted`;
- `k8s.rollout[<slug>]` — one per BentoML service, in rollout order;
- `ec2.deploy[<host>]` / `ec2.verify[<host>]` — one per host (ec2).

In CI: `python3 deploy/deploy.py --target k8s | tail -n 1 > summary.json`. Usage errors
(unknown flag, missing `--target`) come from the argument parser and exit 2 **without** a
JSON summary — parsing precedes the summary machinery.

## config.yml

Every key, with its default. **Everything is optional except `project`, `image` and
`kubernetes.context`/`kubernetes.namespace`.** *Derived* keys cannot be set. Validation
failures exit 2 naming the exact path, e.g. `services.Gateway.expose.node_port`.

| Key | Default | Meaning |
|---|---|---|
| `schema` | *absent = current* | Optional version stamp. Absent or `bentoml-deploy-config/v4`; anything else is rejected with migration instructions naming what moved. |
| `project` | `..` | BentoML project root, **a single path**, resolved relative to the config file itself (`deploy/config.yml`, hence `..`). |
| `build_target` | `null` | `bentoml build` target as `module:Class` (e.g. `service:Gateway`). Only needed when plain `bentoml build` cannot resolve which service to build. |
| `image` | `""` | Image **repository, NO tag** — e.g. `123456789012.dkr.ecr.us-west-1.amazonaws.com/my-bento`, `ghcr.io/acme/my-bento`. `""` or absent pushes nothing (kind/minikube): the image is named after the bento (`<bento>:<version>`) and you load it onto the nodes yourself (`minikube image load` / `kind load docker-image`). Preflight says so every run (`registry.local-image`); nothing outside the cluster can verify a node-local image, and a missing one surfaces as ImagePullBackOff. |
| *image tag* | *derived* | The **bento version**: `--version TAG`, else the project's short git SHA — deterministic and greppable in the cluster. `--image REF` supplies both parts and skips build/push. |
| *registry type* | *derived* | From the `image:` host. `*.dkr.ecr.<region>.amazonaws.com` = ECR: the push logs in with `aws ecr get-login-password` (fresh token per run) and creates the repository if absent, region from the host. Any other host: keep `docker login` valid yourself (`registry.docker-auth`). Empty: nothing pushed. |
| *build platform* | *derived* | Builder architecture (`docker info`) vs. target (`kubectl get nodes -L kubernetes.io/arch`, or each EC2 host's `uname -m`). On a mismatch `bentoml containerize` gets `--opt platform=linux/<target arch>` and says so; docker buildx required and checked. |
| `kubernetes.context` | — | kubectl context; every kubectl call passes `--context` explicitly. |
| `kubernetes.namespace` | — | Must already exist: nothing renders a Namespace object and the script never creates one; preflight fails with the `kubectl create namespace` line. |
| `kubernetes.image_pull_secret` | `null` | Existing secret name, rendered as `imagePullSecrets` on every pod. Preflight checks it exists and, for ECR, warns past 11 h old (tokens expire at 12 h). Refresh by **deleting and recreating** it — `kubectl apply` keeps the original creationTimestamp, so an applied-over secret still looks old. Heuristic: a fresh secret holding an expired token looks fine. |
| `kubernetes.rollout_timeout_seconds` | `900` | Per-Deployment `rollout status` budget. Keep it **above** the startup budget (`probes.startup_failure_threshold` x 10 s = 600 s default): equal, a pod using its whole startup budget loses the race and the rollout fails at the moment it would have succeeded. |
| `kubernetes.local_port` | `3130` | Verification port-forward. `local_port + 1` carries the dependency `/metrics` scrapes; keep both free. |
| `kubernetes.extra_labels` | `{}` | Labels merged onto every rendered object; quoted string values. The `app.kubernetes.io/*` labels are renderer-managed and win (warning if you set one). |
| `kubernetes.manifests_dir` | `null` | **Escape hatch.** When set, rendering is skipped and that directory's YAML is applied **verbatim**, image ref included (see below). |
| `services` | `{}` | **Overrides only.** One block per service you want to change, keyed on the name **exactly** as `bento.yaml` spells it. A key naming no service in the bento is a config error listing the bento's real names (typo protection); a service with no block gets every default below. No `entry:`/`depends:` — they come from the bento, so adding or removing a service is **code-only** (`bentoml.depends(...)` in `service.py`); edit `config.yml` only for non-default resources or exposure. |
| `services.<Name>.slug` | *derived* | Object name and in-cluster hostname: the service name snake_cased, `_`→`-`. Must be a unique DNS-1035 label — set it when the default is not one (a name starting with a digit) or collides. |
| `services.<Name>.replicas` | `1` | Fixed pod count. Omitted from the Deployment when `autoscaling.enabled`. |
| `services.<Name>.resources.requests` | `{cpu: "500m", memory: "1Gi"}` | Quantities as **quoted strings** (`cpu: "2"`, not `cpu: 2`). `@bentoml.service(resources={"cpu","memory"})` is **inert** in open-source BentoML, so these are the only values constraining the pod — mirror the decorator's numbers here to make them bind. A block you write REPLACES the default wholesale. |
| `services.<Name>.resources.limits` | `{cpu: "2", memory: "4Gi"}` | Same rules. GPUs are limits-only: `nvidia.com/gpu: "1"`. |
| `services.<Name>.env` | `{}` | Plain, non-secret env vars; quoted string values. |
| `services.<Name>.env_from_secrets` | `[]` | Existing Secret names → `envFrom.secretRef`. **No static check can see inside a Secret**: a `BENTOML_RUNNER_MAP` (or `BENTOML_SERVE_DEPENDS`) key in one reaches the pod and silently overrides the derived wiring, with no warning anywhere. Keep these Secrets to credentials. |
| `services.<Name>.config_overrides` | `{}` | **Bare** per-service BentoML config (`{workers: 2, traffic: {timeout: 120}}`) — retuning with no image rebuild. Nested under `{"services": {"<Name>": …}}` and emitted as `BENTOML_CONFIG_OVERRIDES` (compact JSON); this per-service form beats the `@bentoml.service(...)` decorator, a global one loses to it. Two traps: it beats the decorator only **under the service's own name**, and a dependency's client timeout comes from the **caller's** config, so raise a dependency's `traffic.timeout` in both blocks. Ignored if you set `BENTOML_CONFIG_OVERRIDES` in `env` (raw wins, with a warning). |
| `services.<Name>.node_selector` | `{}` | Node pinning; quoted string values. Also how you keep a single-arch image scheduled only where it can run on a mixed-architecture cluster. |
| `services.<Name>.tolerations` | `[]` | Verbatim Kubernetes tolerations. |
| `services.<Name>.probes.startup_failure_threshold` | `60` | Startup budget = this x `periodSeconds: 10` (60 → 10 min of model loading). |
| `services.<Name>.probes.readiness_timeout_seconds` | `6` | `timeoutSeconds` on the `/readyz` probes. **6 is a floor, enforced:** BentoML gives a dependency a hard-coded 5 s budget when `/readyz` fans out, so a lower timeout makes the kubelet hang up first and the pod stays permanently unready. Raise for deep chains; never lower. |
| `services.<Name>.autoscaling.enabled` | `false` | `true` renders an HPA **and** omits `spec.replicas` (a declared count and an HPA fight on every apply), from the next deploy on. Back to `false` stops rendering it but does **not** delete the HPA in the cluster: `kubectl delete hpa <slug>`. |
| `services.<Name>.autoscaling.min_replicas` / `max_replicas` | `1` / `5` | HPA bounds (no scale-to-zero). |
| `services.<Name>.autoscaling.metric` / `target` | `cpu` / `70` | `cpu` = `Resource/cpu averageUtilization` (stock metrics-server; requires `requests.cpu`, enforced). `concurrency` = `Pods` metric on `bentoml_service_request_in_progress`, needing a custom-metrics adapter. `target` is % of requested CPU, or in-flight requests per pod. |
| `services.<Name>.expose` | ClusterIP | **Entry service only** — the bento's `entry_service`; rejected elsewhere (exit 2). `type`: `ClusterIP`/`NodePort`/`LoadBalancer`; `annotations` for cloud LBs; `node_port` 30000–32767 and only with `type: NodePort`, for a stable URL — unset (`null`) lets the cluster allocate, and `kubectl apply` preserves it. |
| `services.<Name>.ingress` | disabled | **Entry service only.** `enabled`, `class_name`, `host` (required when enabled), `tls_secret`, `annotations`. |
| `verify.readyz_timeout_seconds` | `600` | How long `/readyz` may take to return 200 after the rollouts finish. |
| `verify.dependency_metrics` | `true` | Prove each dependency served a request (below). Needs `verify.inference`: with no inference block the proof is **skipped** — inapplicable, not disabled, and the value stays `true` in the summary. Set `false` only for a dependency deliberately off the code path of `verify.inference.path`; the run then warns loudly it can no longer tell a working split from an in-process fallback. Needs `local_port + 1` free, `/metrics` on the dependencies, and permission to list pods. |
| `verify.inference.path` / `.body` / `.expect_substring` / `.timeout_seconds` | — / — / `null` / `60` | **Optional** smoke test against the entry service; `body` is native YAML. Omit it and verification is `/readyz` only: pods started and serve HTTP, and a caller reaches its dependencies' `/readyz` — but no real request travels the dependency path. |
| `ec2.*` | — | See the ec2 chapter: `hosts`, `ssh_user`, `ssh_key_path`, `container_name` (defaults to the project directory's name), `host_port`, `env_names`, `registry_auth`, `verify_via`, `local_tunnel_port`. |

Unknown keys are warned about (typo protection), not ignored silently. The two v3
topology keys, `services.<Name>.entry` and `services.<Name>.depends`, are hard errors.

**Older configs.** A v1/v2 `deploy.config.json` or a v3 `config.yml` exits 2 with a
message naming what moved: `entry` → `bento.yaml`'s `entry_service`; `depends` →
`bento.yaml`'s `services[].dependencies`; `image.platform` → auto-detected cross-arch
builds; per-service blocks → optional overrides.

### Where the service topology comes from

The service list, the entry service and the dependency DAG come from the bento's own
`bento.yaml` (`entry_service` + `services[].dependencies`) on every run, from the first
source that answers:

1. **The local bento store**, right after the build stage — `bentoml get <tag> -o json`.
   Authoritative, no docker needed; the normal path.
2. **The image**, when `--skip-build` means no bento is built — `docker run --rm
   --entrypoint cat <image> $BENTO_PATH/bento.yaml`, with `BENTO_PATH` read from the
   image's environment via `docker inspect` (default `/home/bentoml/bento`, but a custom
   base image may move it).
3. **The topology cache** — `deploy/.bento-topology.json`, written beside `config.yml`
   by every successful discovery from 1 or 2. Gitignored by default; **committing it on
   purpose** is the supported way to run `--check-only --local-only` or `--render-only`
   with neither `bentoml` nor docker installed. A cache-sourced run warns that the
   topology is not from this bento: a service added or renamed since would be missing.

When none answers, the run exits 2 naming all three and what to do about each. A run
that builds re-reads the fresh bento afterwards and refreshes the cache, so a stale
cache cannot reach the manifests. `k8s.topology` says which source was used, or that the
topology will come from the build — then `k8s.render` and `k8s.rollout-plan` are skipped,
there being nothing to render yet.

Since the topology is not config input, there is no `depends` to misspell, no cycle to
declare, no second `entry: true`, no `entry: "false"` that YAML reads as true, and no
middle-tier service whose `depends` was forgotten.

### Architecture

A cross-build (see *build platform*) prevents `exec format error`, where every pod
starts, dies instantly and crashloops until the rollout wait calls it at three restarts
(one restart can be a slow dependency or a node under pressure; three is a verdict). On
a **mixed-architecture** cluster no single-arch image satisfies every node: the run warns
and builds for the builder's own arch (or the most common node arch); the fix is a
`node_selector: {"kubernetes.io/arch": …}` per service.

## How the k8s deploy works

**One Deployment + one Service per BentoML service the bento declares**, plus an HPA per
service with `autoscaling.enabled` and one Ingress for the entry service when
`ingress.enabled`. A single-service bento gets exactly one Deployment + Service. Each
run:

1. **Render in memory** from `config.yml` + the discovered topology. The renderer writes
   this run's image ref straight into every Deployment (no image sentinel), derives
   `BENTOML_SERVE_DEPENDS` from the dependency graph, and pipes the set to `kubectl apply
   -n <namespace> -f -`; nothing is written to disk. Applying the full set rather than
   `kubectl set image` reconciles drift and recreates deleted objects. Output is
   deterministic, so `--render-only` diffs are meaningful.
2. **Wait for every rollout, in the derived order** — a topological sort of the
   dependency graph, deepest dependencies first and the entry service last
   (alphabetically within a tier), each with its own `kubectl rollout status
   --timeout=kubernetes.rollout_timeout_seconds`. The order is load-bearing: a caller's
   `/readyz` fans out to its dependencies, so it stays unready (503) until they answer.
   The run **fails fast**: the first failing rollout aborts it, naming the BentoML
   service and Deployment.
3. **Verify through the entry service.** `/readyz` plus the configured inference request
   go through a port-forward to `svc/<entry slug>` on `kubernetes.local_port`, trusted
   only while that process is alive and bound. Only the entry service speaks the public
   HTTP API; dependency Services stay ClusterIP because inter-service payloads are
   `application/vnd.bentoml+pickle`, so exposing one is remote code execution.
4. **Prove the dependencies were actually called** (multi-service bentos,
   `verify.dependency_metrics`, and only with `verify.inference` configured). A green
   `/readyz` and a correct answer do **not** imply the dependency pods did anything: if a
   caller's `BENTOML_SERVE_DEPENDS` does not name a dependency exactly as the bento
   declares it, BentoML instantiates it **in-process** with no log line — that pod loads
   every model itself, answers correctly and reports ready while the dependency pods
   idle, and `/readyz` misses it because it only fans out to remote proxies. So each
   dependency's own `bentoml_service_request_total` is sampled before and after the
   inference request, and verify fails (exit 7) unless it moved. The sample is **per
   pod** — a short-lived port-forward to each Running pod, on `local_port + 1`, summed —
   never through the Service: a Service load-balances, so one scrape reads one random
   replica and a request its sibling served looks like no request at all (a false failure
   on any dependency with `replicas > 1`). If the pods cannot be listed, the run warns
   and falls back to scraping the Service. Health endpoints are excluded from the
   samples, so kubelet probes and the readiness fan-out cannot pass for an inference
   call; scraping `/metrics` is itself uninstrumented.

After a failure, the `k8s.rollout[<slug>]` stages never reached carry `"ok": false` and
`"detail": "not attempted (fail-fast after an earlier service's rollout failed)"`, so CI
can tell "broken" from "never tried".

The wait does not block on `kubectl rollout status` for the full
`rollout_timeout_seconds`: it runs the status watch in short slices, reads the pods in
between, and stops early on two verdicts.

1. **A container state further waiting cannot fix** — `ImagePullBackOff`,
   `ErrImagePull`, `InvalidImageName`, `ImageInspectError`, `RegistryUnavailable`,
   `CreateContainerConfigError`, `CreateContainerError`, or `CrashLoopBackOff` past
   three restarts. The error carries kubelet's own message: `cannot succeed: pod <name>:
   ImagePullBackOff — …`. Missing pull credentials and a wrong image reference fail in
   seconds instead of stalling CI for the whole timeout.
2. **The Deployment's own `progressDeadlineSeconds` expiring** (10 minutes by default) —
   Kubernetes itself has stopped trying. The run stops and quotes what the pods say:
   `the Deployment exceeded its own progress deadline (pod …: Unschedulable — 0/1 nodes
   are available: 1 Insufficient cpu…)`. Capacity: the default requests are 500m CPU /
   1Gi **per service**, so a four-service bento needs 2 CPU and 4Gi of *schedulable*
   room, times `replicas`.

A merely slow pod (a large image, a model download) is never interrupted: only those two
verdicts cut the wait short, and `rollout_timeout_seconds` still bounds everything else
— if it is what expires, kubectl's last line is logged, not swallowed. Pod inspection is
read-only and any kubectl failure is ignored, so it can only shorten a failure that was
going to happen anyway.

### Review before applying: `--render-only`

```bash
python3 deploy/deploy.py --target k8s --render-only            # -> deploy/rendered/
python3 deploy/deploy.py --target k8s --render-only /tmp/next  # -> any directory
python3 deploy/deploy.py --target k8s --render-only --version review   # no git needed
```

**It needs an image reference and a topology.** The reference: `--image REF`, `--version
TAG` with `image:`, or a git checkout (default tag `git rev-parse --short HEAD` of
`project`); outside a git repo the run exits 2 telling you to pass one. The topology: the
image or the `.bento-topology.json` cache, since nothing is built here — a successful
discovery refreshes that cache, seeding it for later build-free runs.

It writes one file per object (`<slug>-deployment.yaml`, `<slug>-service.yaml`,
`<slug>-hpa.yaml`, `ingress.yaml`), prints the rollout order it derived, and exits 0
without contacting the cluster, building or applying anything. Use it to:

- **review a config change**: render before and after and `diff -r` the two directories
  (output is deterministic — stable key order, sorted maps);
- **dry-run against the API server**: `kubectl --context {{K8S_CONTEXT}} apply
  --dry-run=server -f deploy/rendered`;
- **feed GitOps**: commit the rendered directory, or hand it to Argo CD/Flux. The
  Deployments carry a **concrete** image ref (the resolved tag), so re-render on every
  release, or pass `--image`/`--version` explicitly.

`deploy/rendered/` is generated output: safe to delete, re-created on demand, never read
by the deploy path — `.gitignore` it unless you are deliberately committing manifests for
GitOps. Re-rendering **prunes** its own stale files: a `<slug>-deployment.yaml`,
`<slug>-service.yaml`, `<slug>-hpa.yaml` or `ingress.yaml` that no longer matches the
config (renamed slug, deleted service, autoscaling off) is deleted and the removal
logged — left behind, `kubectl apply -f <dir>` or a GitOps controller would re-create the
orphaned Deployment and Service and its pods would keep serving. Files the renderer does
not own (a sidecar patch, a kustomization, a README) are never touched, but they *are*
listed with a warning, because applying the directory applies them too.

Every rendered object carries `app.kubernetes.io/name: <slug>` (also the pod selector),
`app.kubernetes.io/component: <BentoML service name>`, `app.kubernetes.io/part-of:
<bento name>` and `app.kubernetes.io/managed-by: bentoml-k8s-deploy`, plus
`kubernetes.extra_labels`. The interactive `bentoml-k8s-deploy` skill renders the same
file names, shapes and `managed-by` value from the same `config.yml`, so `kubectl get all
-l app.kubernetes.io/managed-by=bentoml-k8s-deploy` finds the whole deployment however it
was applied, and switching between the two rewrites nothing.

### Escape hatch: `kubernetes.manifests_dir`

Set it (a path relative to `deploy/`) when you have genuinely outgrown `config.yml` — a
sidecar, a PVC, a ServiceMonitor, an init container. Then:

- rendering is **skipped**; every `*.yaml`/`*.yml` in that directory is applied
  **verbatim**, in name order, in one pass. Nothing is substituted — **including the
  image reference**: those files pin whatever tag they carry, so this run's built/pushed
  ref does not reach the pods unless the files name it. The deploy log and the
  `k8s.apply` stage's `detail` say so every run, and preflight warns when the pinned
  refs differ from the resolved image;
- the bento's topology still drives the rollout order, the verification target and the
  wiring checks (so `--skip-build` still needs docker or the cache) — but `config.yml` no
  longer drives the workload shape;
- `--render-only` refuses to run (nothing to render) and says so.

Migration path: `--render-only DIR`, commit `DIR`, then point `kubernetes.manifests_dir`
at it; from then on edit the tag per build. To re-render later: unset the key, render,
set it back, re-apply your own edits. The trade-off: *you* own the labels, probes,
dependency wiring, namespaces **and image tags** in that YAML, and adding a BentoML
service means editing both the config and the manifests.

This is also the answer for the knobs `config.yml` does not have — `path_prefix` probe
paths, `terminationGracePeriodSeconds`, liveness timings, `imagePullPolicy`,
`revisionHistoryLimit`, HPA `behavior` windows, affinity/`topologySpreadConstraints`,
volumes, sidecars, PodDisruptionBudgets.

### What preflight checks

Preflight runs as **one aggregated pass**: every applicable check runs, and a single
exit 3 reports all failures together.

Credential-free (so `--check-only --local-only` runs them): `common.pyyaml`,
`common.python-version`, `common.git-repo`, `build.*`, `k8s.kubectl-cli`, plus

- `k8s.topology` — which source answered, the bento tag, the service list and the
  derived rollout order. A hard failure under `--skip-build` (nothing later will produce
  it), a warning otherwise;
- `k8s.render` — renders the manifests exactly as the deploy would and parses every
  document, so a config that cannot render fails before anything is built;
- `k8s.rollout-plan` — the derived rollout order and every `BENTOML_SERVE_DEPENDS` value.

Needing the cluster: `k8s.context-reachable`, `k8s.rbac-create-deployment`,
`k8s.namespace`, `k8s.image-pull-secret`, and `k8s.node-arch` (reads the nodes'
`kubernetes.io/arch` label and, for a cross-build, requires docker buildx).

Checks on hand-written YAML and config-declared topology are gone because their input
is: the image ref, `metadata.namespace`, `BENTOML_SERVE_DEPENDS`, non-entry ClusterIP and
the omission of `replicas` beside an HPA all come from the renderer; `entry`, `depends`,
cycles and missing service blocks are not config input. (The tier sort is still guarded
against a corrupted `bento.yaml` or hand-edited cache, reported as a bento problem.)

**A runner map is emitted by nothing, ever**, and a `BENTOML_RUNNER_MAP` in `env` is a
config error. The one supported way to take over the wiring is `env:
{BENTOML_SERVE_DEPENDS: …}`, which **replaces** the derived value; the loader applies the
same rules to your string (one `=` per pair, no commas, http(s) URLs, every dependency
the bento declares covered — a misspelled name fails with a path) and `--check-only`
prints the raw value marked `[raw]`, not the derived one.

The manifest-shape checks stay **fully active** when `kubernetes.manifests_dir` is set —
that YAML is hand-owned, so `k8s.manifests`, `k8s.serve-depends` and
`k8s.service-exposure` check namespace consistency, dependency wiring, no runner map,
non-entry ClusterIP, no `replicas:` beside an HPA, an `image:` line in every Deployment
(verbatim, so a drifting tag is a warning naming both refs), one
`<slug>-deployment.yaml` + `<slug>-service.yaml` per configured service, and no leftover
`{{PLACEHOLDER}}`.

The rest of the invariants live in config validation, where the error can name your
line: unknown `services:` keys, a `services:` key naming no service of the bento (the
typo guard, exit 2, message listing the bento's real names), unique DNS-1035 slugs,
quantities that are quoted strings, a NodePort in range and only with `type: NodePort`,
`expose`/`ingress` only on the entry service, the readiness-timeout floor, an `image:`
with no tag and no digest, a `project:` that exists, no duplicate YAML keys (which YAML
itself resolves last-wins in silence), a `manifests_dir` that exists, JSON-serializable
`verify.inference.body` / `config_overrides` / `tolerations` (checked at load, not when
the smoke test POSTs after the cluster was already changed), and no runner-map env var.
One hole no check can close: `env_from_secrets` mounts whole Secrets, and a
`BENTOML_RUNNER_MAP` key inside one reaches the pod unseen.

### Known upstream noise

`bentoml containerize` (1.4.x) emits a few `UserWarning: The parameter --quiet is used
more than once` lines from click on every invocation — an upstream duplicate-option
declaration, not a problem with this bundle or your project. It lands in the build
stage's stderr; ignore it.

## How the ec2 deploy works

`--target ec2` deploys the image onto the **existing** instances in `ec2.hosts` — the
script never provisions EC2 (use the interactive `bentoml-ec2-deploy` skill first). Hosts
are deployed and verified sequentially and **fail fast**: the first failing host aborts
the run, and unattempted hosts are marked in the `ec2.deploy[<host>]` /
`ec2.verify[<host>]` stages. Every remote docker command runs under `sudo` over key-based
SSH (`ec2.ssh_key_path`, BatchMode — nothing may prompt). Per host:

1. **Registry login** — with `registry_auth: "ecr-token-over-ssh"`, the local AWS
   credentials mint a fresh ECR token (`aws ecr get-login-password`, region from the
   image ref) and pipe it over the SSH channel's stdin into `sudo docker login
   --password-stdin`; the token never touches a file or a command line. With
   `"preauthed"` the instances must already be able to pull (instance profile, or a
   `docker login` you keep fresh yourself — ECR tokens expire after 12 h).
2. `sudo docker pull <image>`
3. `sudo docker rm -f <container_name>` (idempotent redeploy), then `sudo docker run -d
   --name <container_name> --restart unless-stopped -p <host_port>:3000 ... <image>`
4. Wait 3 s, then require `docker ps` to report the container **Up** — a crash-looping
   container still shows `Up` at t=0, so the wait is load-bearing.
5. **Verify** `/readyz` plus the configured inference request:
   - `verify_via: "tunnel"` (default) — a hardened SSH tunnel (`-o
     ExitOnForwardFailure=yes`, liveness-checked around every probe so a local port
     squatter cannot fake a green result) on local port `ec2.local_tunnel_port`. Works
     even when the security group never exposes the host port.
   - `verify_via: "direct"` — plain HTTP to `http://<host>:<host_port>`; needs the
     security group to allow that port from the deploying machine. The container is
     re-checked over SSH first so the responder is known to be ours.

The ec2 target renders no manifests, needs no topology, and ignores `services:` /
`kubernetes:`. Architecture matching needs no config: each host's `uname -m` is compared
with the builder's and the image is cross-built when they differ
(`ec2.remote-arch[<host>]`). All hosts must share one architecture — a single image
cannot satisfy two, and preflight fails naming the hosts and their arches.

`ec2.env_names` lists variable **names only** (e.g. `["HF_TOKEN"]`). At deploy time each
value is read from the deploying process's environment — preflight fails if any is
missing — sent base64-encoded over the SSH channel's stdin, exported on the remote side,
and handed to the container with docker's value-less `-e NAME` form. Values never appear
in the config, in any file, or in any command line on either machine (`ps` shows nothing
on either end).

**Rollback (ec2).** `docker rm -f` + `run` replaces the container in place, so there is
no previous revision to undo into — a rollback is a redeploy of the previous tag
(deterministic git-SHA tags make this easy). Exit codes and the summary contract are
unchanged.

```bash
python3 deploy/deploy.py --target ec2 --skip-build --image {{IMAGE_URL}}:<previous-tag>
```

## Rollback (k8s)

Kubernetes keeps the previous ReplicaSets **per Deployment**, so roll back every service
the run touched, dependencies first and the entry service last (the deploy order, for
the same reason). One pair per service:

```bash
# dependencies first:
kubectl --context {{K8S_CONTEXT}} -n {{NAMESPACE}} rollout undo deployment/{{DEP_SERVICE_SLUG}}
kubectl --context {{K8S_CONTEXT}} -n {{NAMESPACE}} rollout status deployment/{{DEP_SERVICE_SLUG}}
# entry service last:
kubectl --context {{K8S_CONTEXT}} -n {{NAMESPACE}} rollout undo deployment/{{ENTRY_SERVICE_SLUG}}
kubectl --context {{K8S_CONTEXT}} -n {{NAMESPACE}} rollout status deployment/{{ENTRY_SERVICE_SLUG}}
```

A failed deploy prints exactly this list for your services, plus the `get pods` /
`describe` / `logs` commands for each one, in its error hint. Or redeploy a known-good
tag explicitly:

```bash
python3 deploy/deploy.py --target k8s --skip-build --image {{IMAGE_URL}}:<previous-tag>
```

A rollback of the **image** is not a rollback of the **config**: the manifests are
rendered from `config.yml` at your current checkout, so revert the config change too —
one more reason to commit `config.yml` and keep `--render-only` diffs in review.

## CI/CD

The bundle can be the *entire* deploy step of a pipeline: exit codes 0..7 tell CI what
failed, and the last stdout line is the JSON summary (`... | tail -n 1 > summary.json`,
or `--output-json`). Three patterns cover everything:

- **PR gate**: `--check-only --local-only` — config validity, rendering, tool presence.
  Needs **no** credentials, cluster or registry, so it is fork-PR safe — but it does check
  that the target's CLI tools are **on PATH** (docker always; the AWS CLI v2 for ECR;
  kubectl for k8s; ssh for ec2), so the job must install them. Rendering needs the
  topology: with the image absent locally, commit `deploy/.bento-topology.json`, else
  `k8s.render` and `k8s.rollout-plan` are skipped and the gate covers config validity
  only.
- **Manifest diff in review** (optional, k8s): `--render-only` on the PR and on the base
  commit, then `diff -r` the two directories and post the result.
- **Deploy on main**: the full run. The runner needs Python >= 3.9, PyYAML, the `bentoml`
  CLI, docker (build/containerize/push), the AWS CLI v2 for ECR / EC2-with-ECR, and
  target-specific access (kubeconfig, SSH key, or an AWS role).

Any job that does **not** `pip install bentoml` (a `--skip-build` deploy, a render-only
diff job) must `pip install pyyaml` — that is the config parser. Such a job has no bento
either, so its topology comes from the image (docker required) or a **committed**
`deploy/.bento-topology.json`; re-commit that file when the bento's service set changes.

Secrets wiring (nothing is ever read from the config or written into the repo):

| Secret | Used by | How it reaches the script |
|---|---|---|
| AWS role (OIDC, preferred) or access keys | ECR push; ec2 with `ecr-token-over-ssh` | ambient AWS env — `aws-actions/configure-aws-credentials` on GitHub, `AWS_ROLE_ARN` + `AWS_WEB_IDENTITY_TOKEN_FILE` on GitLab |
| SSH private key | ec2 | written by the job to the path in `ec2.ssh_key_path`, mode 600 |
| Runtime env values (names in `ec2.env_names`) | ec2 | exported in the job's environment |

### GitHub Actions

A complete workflow (`.github/workflows/deploy.yml`). Keep only the deploy job(s) for
the target(s) generated into this bundle. The AWS jobs use OIDC (no long-lived keys):
create one IAM role trusted by GitHub's OIDC provider and store its ARN in the
`AWS_DEPLOY_ROLE_ARN` repository secret.

```yaml
name: deploy
on:
  pull_request:
  push:
    branches: [main]
permissions:
  contents: read
jobs:
  # PR gate: no credentials, no cluster/registry connectivity — fork-safe.
  check:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install "bentoml>=1.4"     # brings PyYAML with it
      - run: python3 deploy/deploy.py --target k8s --check-only --local-only
      # one line per generated target:
      # - run: python3 deploy/deploy.py --target ec2 --check-only --local-only

  # Optional: show reviewers the manifest diff a config.yml change produces.
  render-diff:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: {fetch-depth: 0}
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install pyyaml            # no bentoml needed to render
      - run: python3 deploy/deploy.py --target k8s --render-only /tmp/head --version ci
      - run: git checkout ${{ github.event.pull_request.base.sha }}
      - run: python3 deploy/deploy.py --target k8s --render-only /tmp/base --version ci
      - run: diff -ru /tmp/base /tmp/head || true

  # k8s target: OIDC -> AWS credentials -> ECR push + EKS kubeconfig.
  deploy-k8s:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions: {id-token: write, contents: read}   # id-token for configure-aws-credentials
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install "bentoml>=1.4"
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_DEPLOY_ROLE_ARN }}
          aws-region: <ecr-region>
      # the --alias must equal kubernetes.context:
      - run: aws eks update-kubeconfig --name <cluster> --alias {{K8S_CONTEXT}}
      - run: python3 deploy/deploy.py --target k8s --output-json summary.json
      - uses: actions/upload-artifact@v4
        if: always()
        with: {name: deploy-summary-k8s, path: summary.json}

  # ec2 target: a plain SSH-key secret; AWS credentials are only needed for ECR
  # images (push, and registry_auth "ecr-token-over-ssh").
  deploy-ec2:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions: {id-token: write, contents: read}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install "bentoml>=1.4"
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_DEPLOY_ROLE_ARN }}
          aws-region: <ecr-region>
      - name: Install the SSH key   # exact path in ec2.ssh_key_path; preflight enforces 600
        env:
          EC2_SSH_KEY: ${{ secrets.EC2_SSH_KEY }}
        run: |
          mkdir -p ~/.ssh
          printf '%s\n' "$EC2_SSH_KEY" > ~/.ssh/deploy_key.pem
          chmod 600 ~/.ssh/deploy_key.pem
      - run: python3 deploy/deploy.py --target ec2 --output-json summary.json
        # runtime env vars named in ec2.env_names go here, e.g.:
        # env:
        #   HF_TOKEN: ${{ secrets.HF_TOKEN }}
      - uses: actions/upload-artifact@v4
        if: always()
        with: {name: deploy-summary-ec2, path: summary.json}
```

The jobs install no CLI tools because `ubuntu-latest` runners already ship docker, the
AWS CLI v2 and kubectl. On self-hosted runners, install whatever preflight reports
missing (exit 3 names the tool). Fork PRs never reach the deploy jobs (`push` to `main`
only), and the PR gate needs no secrets. A trusted-PR "full preflight" variant is the
same deploy job with `--check-only` substituted in.

### GitLab CI

The equivalent `.gitlab-ci.yml`, using GitLab's OIDC (`id_tokens`) — the AWS CLI picks
up `AWS_ROLE_ARN` + `AWS_WEB_IDENTITY_TOKEN_FILE` automatically. The deploy job needs a
docker daemon (docker-in-docker below, or a runner with a mounted socket) and installs
the AWS CLI v2 itself (v2 is not pip-installable).

```yaml
stages: [check, deploy]

# --local-only skips connectivity checks but still verifies tool PRESENCE, so the bare
# python image needs the target CLIs: docker (the CLI alone satisfies the check), the AWS
# CLI v2 for ECR images, kubectl for k8s (git and ssh already ship in python:3.11).
.install_tools: &install_tools
  - apt-get update -qq && apt-get install -y -qq curl unzip docker.io
  - curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  - unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install
  - curl -sSLo /usr/local/bin/kubectl "https://dl.k8s.io/release/$(curl -sSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  - chmod +x /usr/local/bin/kubectl
  - pip install "bentoml>=1.4"      # brings PyYAML with it

check:
  stage: check
  image: python:3.11
  rules: [if: '$CI_PIPELINE_SOURCE == "merge_request_event"']
  before_script: *install_tools
  script:
    - python3 deploy/deploy.py --target k8s --check-only --local-only
    # one line per generated target:
    # - python3 deploy/deploy.py --target ec2 --check-only --local-only

deploy-k8s:
  stage: deploy
  image: python:3.11
  rules: [if: '$CI_COMMIT_BRANCH == "main"']
  services: [docker:27-dind]
  id_tokens:
    GITLAB_OIDC_TOKEN: {aud: "https://gitlab.com"}
  variables:
    DOCKER_HOST: tcp://docker:2375
    DOCKER_TLS_CERTDIR: ""
    AWS_ROLE_ARN: arn:aws:iam::<account>:role/<gitlab-deploy-role>
    AWS_WEB_IDENTITY_TOKEN_FILE: /tmp/web-identity-token
  before_script:
    - echo "$GITLAB_OIDC_TOKEN" > "$AWS_WEB_IDENTITY_TOKEN_FILE"
    - *install_tools
    # the --alias must equal kubernetes.context:
    - aws eks update-kubeconfig --name <cluster> --alias {{K8S_CONTEXT}}
  script:
    - python3 deploy/deploy.py --target k8s --output-json summary.json
  artifacts:
    when: always
    paths: [summary.json]
```

A deploy job that skips the build (`--skip-build --image <ref>`) does not need
`bentoml`, docker or dind — but it **does** need `pip install pyyaml`.

For the ec2 target, swap the last `script:` line for `--target ec2`, drop the kubectl
install and the `aws eks update-kubeconfig` line, write the SSH key from a **file-type**
CI/CD variable to the path in `ec2.ssh_key_path` with mode 600 in `before_script`, and
define any `ec2.env_names` values as masked CI/CD variables (AWS credentials stay needed
only for ECR images — push, and `registry_auth: "ecr-token-over-ssh"`).

For the k8s target on a non-EKS cluster pulling from ECR, the image pull secret expires
every 12 h — delete and recreate it in CI before deploying (this applies regardless of
which targets this bundle carries, so keep it when pruning).
