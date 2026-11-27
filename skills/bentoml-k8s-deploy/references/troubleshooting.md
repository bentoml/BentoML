# Troubleshooting a deployed bento

Read this when a rollout fails or a deployed service misbehaves. Everything here is
per **BentoML service**: one Deployment + Service each, so run the commands for the
service you are diagnosing. `<ns>` is the namespace, `<name>` the service's slug
(= its object name and its `app.kubernetes.io/name` label).

**Diagnose the deepest dependency first.** A caller's `/readyz` fans out to its
dependencies, so an entry service stays 503 purely because something two tiers down is
broken.

```sh
kubectl get deploy,pods -n <ns> -L app.kubernetes.io/component
kubectl describe pod -n <ns> <pod>            # Events, at the bottom, first
kubectl logs -n <ns> <pod> --tail=100
kubectl logs -n <ns> <pod> --previous --tail=100   # if it restarted
```

Labels: `component` = BentoML service name, `name` = slug, `part-of` = bento.

**Fixes are config edits, not `kubectl edit`.** The manifests are rendered from
`deploy/config.yml` on every run, so change the config and re-run:

```sh
python3 deploy/deploy.py --target k8s --render-only /tmp/fix           # see the change
python3 deploy/deploy.py --target k8s --skip-build --image <current>   # apply it
```

`kubectl edit`/`patch` is fine for testing a hypothesis — say out loud that the next
run reverts it. Projects that set `kubernetes.manifests_dir` own their YAML and edit it
directly. Never `kubectl delete` anything this session did not create.

| Symptom | Section |
|---|---|
| `ImagePullBackOff` / `ErrImagePull` | 1 |
| `CrashLoopBackOff` / `Error` | 2 |
| `OOMKilled` in describe's `Last State` | 3 |
| `Pending` | 4 |
| `Running` but `0/1` ready | 5 |
| Rollout never completes | 6 |
| Pods ready, service unreachable | 7 |
| Reachable, but inference 4xx/5xx | 8 |
| `kubectl apply` rejected an object | 9 |
| Everything green, dependency pods idle | 10 |

## 1. ImagePullBackOff / ErrImagePull

```sh
kubectl describe pod -n <ns> <pod> | grep -A5 -i "failed to pull"
kubectl get pod -n <ns> <pod> -o jsonpath='{.spec.imagePullSecrets}{"\n"}'
```

- **401 / `unauthorized`** — no credentials in the namespace. Either set
  `kubernetes.image_pull_secret`, or check whether the `default` serviceaccount already
  carries one (`kubectl -n <ns> get sa default -o jsonpath='{.imagePullSecrets[*].name}'`).
  ECR tokens expire after **12 h**, so an old secret fails like a missing one — delete and
  recreate it (`kubectl apply` over a secret keeps its creationTimestamp, so it still
  looks stale).
- **`not found` / `manifest unknown`** — that tag was never pushed. The ref is
  `<image URL>:<bento version>`; push it, or deploy an existing tag with
  `--skip-build --image <ref>` (preflight verifies it exists first).
- **`no match for platform`** — image built for the wrong arch. A build run detects this
  (`k8s.node-arch`); a hand-built image needs
  `bentoml containerize --opt platform=linux/amd64`.
- **kind/minikube** — local clusters cannot see your Docker images. `kind load
  docker-image <ref> --name <cluster>` or `minikube image load <ref>`. An empty `image:`
  in the config is exactly this case: nothing is pushed, the image is named after the
  bento.

## 2. CrashLoopBackOff

The answer is in the *previous* attempt's logs; read the tail bottom-up.

```sh
kubectl logs -n <ns> <pod> --previous --tail=200
kubectl get pod -n <ns> <pod> -o jsonpath='{.spec.containers[0].env}{"\n"}'
```

- **Traceback about a missing env var** — create the Secret (ask the user for the value,
  never invent one) and reference it from `services.<Name>.env_from_secrets`.
- **`NotFound: model ...` or an HF download error** — the model is not baked in. Either
  rebuild the bento with it, or give the pod `HF_TOKEN` and egress.
- **`ModuleNotFoundError`** — dependency missing from `bentofile.yaml` /
  `bentoml.images.Image(...)`. Rebuild, re-containerize, push.
- **`exec format error`** — wrong arch (section 1).
- **Exit 137, no traceback** — OOMKilled (section 3).
- Never "fix" a crash by adding `command:` to the pod spec. The entrypoint runs `serve`
  and activates the venv; overriding it breaks the container.

## 3. OOMKilled

Memory limit below model weights + working set. Rule of thumb: limit ≥ 1.5–2× the
weights on disk. Raise `services.<Name>.resources.limits.memory` (and `requests`) in
`config.yml` — quantities are **quoted strings** — and re-run. If the node itself is too
small this becomes section 4. An OOM during startup looks like a probe failure but is not.

## 4. Pending

```sh
kubectl describe pod -n <ns> <pod> | grep -A10 Events
kubectl describe nodes | grep -A6 "Allocated resources"
```

- **`Insufficient cpu` / `Insufficient memory`** — do the arithmetic before blaming the
  cluster: the default request is **500m CPU / 1Gi per service**, so a four-service bento
  needs 2 CPU and 4Gi of *schedulable* room, times `replicas`. A node can sit at 98%
  requested CPU while `kubectl top node` shows 40% used — scheduling counts requests, not
  usage. Lower `resources.requests` or add nodes. A deploy run reports this itself once
  the Deployment's `progressDeadlineSeconds` expires, naming the pod's `Unschedulable`
  message.
- **`Insufficient nvidia.com/gpu`** — no GPU nodes, or the NVIDIA device plugin is not
  installed so GPUs are not advertised. Installing it is the cluster admin's job — say so
  rather than touching the cluster.

  ```sh
  kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
  ```
- **`didn't match node affinity/selector`** or **`untolerated taint`** — fix
  `services.<Name>.node_selector` / `tolerations` against `kubectl get nodes --show-labels`.

## 5. Running but not Ready

Decide *still loading* vs *broken*.

```sh
kubectl describe pod -n <ns> <pod> | grep -iA4 "unhealthy\|probe failed"
kubectl logs -n <ns> <pod> --tail=50
```

Model loading in the logs and a pod only minutes old: **wait**. The startup budget is
`periodSeconds: 10` × `failureThreshold: 60` ≈ 10 min. If loading truly needs longer,
raise `services.<Name>.probes.startup_failure_threshold` — never shrink or drop the probe
— and keep `kubernetes.rollout_timeout_seconds` above the new budget, or a pod that just
makes it loses the race.

Otherwise ask `/readyz` yourself and **read the body**; BentoML puts the reason there.
Run the tunnel and the curls in **one shell invocation** (`%1` does not survive across
tool calls) and wait for the bind:

```sh
kubectl port-forward -n <ns> <pod> 3100:3000 &
for i in $(seq 1 10); do curl -so /dev/null http://localhost:3100/livez && break; sleep 1; done
curl -si http://localhost:3100/readyz
kill %1
```

- `200`, empty body → ready. If Kubernetes still reports failure, the probe path/port is
  wrong (must be `/readyz` on 3000).
- `503 ... __is_ready__() returns False` → the user's own hook; read their `service.py`.
- `503 Runners are not ready` → a dependency is not up; diagnose that service.

## 6. Rollout never completes

New-ReplicaSet pods never became Ready — diagnose those pods with sections 1–5. A deploy
run stops on its own for terminal states (`ImagePullBackOff`, `CreateContainerConfigError`,
`CrashLoopBackOff` past three restarts) and when the Deployment's own progress deadline
expires, quoting kubelet. To get back to a good state, redeploy the last good ref
(`--skip-build --image <previous>`), which restores every service in dependency order;
`kubectl rollout undo deployment/<name> -n <ns>` is also fine (it deletes nothing).
Never re-push a tag with new content — nodes cache by ref, so use a new tag.

## 7. Pods ready, service unreachable

```sh
kubectl get endpointslices -n <ns> -l kubernetes.io/service-name=<name>
kubectl get pods -n <ns> -l app.kubernetes.io/name=<name> --show-labels
```

- **No endpoints** — selector does not match, or pods are not Ready (sections 2–5). The
  renderer puts `app.kubernetes.io/name: <slug>` on both sides and wins over
  `extra_labels`, so a mismatch means hand-edited YAML, a stale Deployment from an older
  naming scheme, or a `manifests_dir` project whose selector and template drifted.
- **Endpoints exist, curl fails** — port wiring. `targetPort` must be `3000`. Bisect:
  port-forward the pod directly, then the Service.
- **NodePort** — needs the node's reachable IP plus a firewall/security-group rule for
  30000–32767. **LoadBalancer** stuck `<pending>` means no cloud controller.
  **Ingress** with an empty ADDRESS usually means no controller for that
  `ingressClassName`; installing one is the user's call.

## 8. Reachable, but inference 4xx/5xx

Never guess the payload — read the schema the server publishes:

```sh
kubectl port-forward -n <ns> svc/<name> 3100:3000 &
for i in $(seq 1 10); do curl -so /dev/null http://localhost:3100/livez && break; sleep 1; done
curl -s http://localhost:3100/docs.json | python3 -m json.tool | head -60
curl -si -X POST http://localhost:3100/<endpoint> -H 'Content-Type: application/json' -d '{"<arg>": "<value>"}'
kill %1
```

- **404** — wrong path; endpoints are `POST /<method_name>`, listed under `paths`.
- **400 / 422** — body does not match the method signature; keys are the Python parameter
  names.
- **405** — inference routes are POST-only.
- **500** — server-side exception; the traceback is in the pod log at that moment.
- **503 / 429** — overloaded or at the concurrency limit. Raise
  `services.<Name>.config_overrides.traffic.timeout` if it is a timeout — and note a
  dependency's client timeout is read from the **caller's** config, so set it in both.

## 9. `kubectl apply` rejected an object

Objects are applied one at a time, so a rejection can leave a half-applied state — list
what exists before fixing. Usual cause is an invalid name: Service names are DNS-1035
(must start with a letter). The renderer validates slugs at config load time (exit 2,
before anything is applied), so this points at a `services.<Name>.slug` you can fix, or
at hand-owned `manifests_dir` YAML.

## 10. Everything green, dependency pods idle

Multi-service bentos only, and the dangerous one: every pod Ready, the entry service
returns **correct** answers, and the dependency pods serve nothing.

Cause: the entry pod could not resolve a dependency's URL and instantiated it
**in-process**. BentoML falls back silently, and `/readyz` only fans out to dependencies
it holds a remote proxy for — so readiness is trivially true. You get a single-pod bento
with a multi-pod bill: the dependency's replicas, resources and HPA are inert, and one pod
loads every model (a frequent OOM cause).

Prove it with the dependency's own counter — scrape, send **one** request to the entry
service, scrape again:

```sh
kubectl port-forward -n <ns> svc/<dependency> 3101:3000 &
curl -s localhost:3101/metrics | grep '^bentoml_service_request_total'
# one real request to the ENTRY service here, then:
curl -s localhost:3101/metrics | grep '^bentoml_service_request_total'
kill %1
```

Unchanged (ignoring `/livez`, `/readyz`, `/metrics` samples — kubelet probes move them
too) means the call never crossed the network. With `verify.inference` configured,
`deploy.py` runs exactly this check after every deploy and fails with **exit 7**; on a
dependency with several replicas it sums all of its pods, because a Service scrape reads
one replica at random.

Then look at the caller's environment:

```sh
kubectl get deploy -n <ns> <entry> -o jsonpath='{.spec.template.spec.containers[0].env}' | tr ',' '\n'
```

- **`BENTOML_SERVE_DEPENDS` missing a dependency** — every service that has dependencies
  needs it, middle tiers included. The renderer derives it from `bento.yaml`, so re-running
  fixes it; a `manifests_dir` project must list every dependency as
  `Name=http://<slug>.<ns>.svc.cluster.local:3000`, whitespace-separated, names spelled
  exactly as `bento.yaml` does.
- **`BENTOML_RUNNER_MAP` present** — remove it. The serving parent overwrites it and any
  dependency missing from the result is instantiated in-process. `config.yml` rejects it;
  hand-written YAML does not.
- **`command:` overridden** — plain `bentoml serve` runs the whole DAG in one pod by
  design. The entrypoint must stay in charge (`start-http-server --service-name <Name>`).

## Reading BentoML logs

- Healthy startup ends with `Starting production HTTP BentoServer from "<id>" listening on
  http://0.0.0.0:3000`. If that line is present, the server bound the port — later
  failures are probes, networking or per-request errors, not startup.
- Before it: worker and model-loading output. No startup line and logs stopping mid-load
  means still loading, a crash during load (traceback), or OOM (cut off, exit 137).
- Access logs print one line per request — useful to confirm requests arrive at all.
  `/livez` and `/readyz` hits are excluded, so their absence is normal.
