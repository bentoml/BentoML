---
name: bentoml-deploy-scriptgen
description: >
  Generate a standalone, committable production deploy-script bundle
  (deploy/deploy.py + one config.yml — overrides only — from which the
  Kubernetes manifests are rendered) that builds, containerizes, pushes, deploys, and
  verifies a BentoML service without any agent involved — runnable from a
  terminal or CI/CD. Use when the user says things like "generate a deployment
  script", "deploy from CI/CD", "set up a production deployment pipeline",
  "deploy without the agent", "give me a script I can commit to deploy this",
  or "automate my BentoML deploys". Complements the interactive skills:
  bentoml-containerize, bentoml-k8s-deploy, and bentoml-ec2-deploy do a one-off
  deploy with you in the loop; this skill emits scripts that repeat it forever.
  Kubernetes and EC2 targets.
license: Apache-2.0
compatibility: >-
  Emits Python >= 3.9 scripts whose only third-party dependency is PyYAML; at deploy time they call the bentoml, docker and kubectl CLIs, plus the AWS CLI for ECR or EC2.
---

# Generate a production deploy-script bundle

You are a **script generator, not a script author**: the `.py` files under this
skill's `templates/deploy/` are reviewed, e2e-tested artifacts.

**HARD RULE — copy the `.py` files VERBATIM. Never edit, patch, "improve" or
regenerate their code, and never write new script logic.** Two rendered files
carry everything user-specific: `config.yml` and `README.md`. No manifests to
write — the bundle renders them from `config.yml` on every run. If a need cannot
be expressed in `config.yml`, say so, point at `kubernetes.manifests_dir` (the
documented escape hatch), and report it as a limitation. Never fork the
templates.

This bundle is also the interactive path's renderer and validator:
`bentoml-k8s-deploy` copies `templates/deploy/` with the Step 2 commands, then
calls `deploy.py --target k8s --render-only` (and `--check-only`).
`_internal/render.py` + `_internal/config.py` are the single source of truth for
both flows; no second renderer exists. Two config layouts, both exercised:

- **bundle** (this skill): `deploy/config.yml` with `project: ..`;
  `python3 deploy/deploy.py --target k8s [--render-only]` finds it beside
  `deploy.py`.
- **root-level**: `config.yml` at the project root with `project: .`, passed
  explicitly:
  `python3 deploy/deploy.py --target k8s --render-only k8s --config config.yml`.

Relative paths (`project`, `kubernetes.manifests_dir`) resolve against **the
config file's own directory**; both layouts behave identically.

**config.yml v4 is overrides-only**: no service list, no `entry:` flag, no
`depends:`. Service list, entry service and dependency DAG come from the bento's
own `bento.yaml` at RUN time — the local bento store after the build,
`docker run --rm --entrypoint cat <image> $BENTO_PATH/bento.yaml` under
`--skip-build`, or the `deploy/.bento-topology.json` cache. Nothing to keep in
sync with the code. Image tag = bento version; ECR recognized from the image URL's
host; cross-architecture builds auto-detected. **Four values are enough to
deploy**: `project`, `image`, `kubernetes.context`, `kubernetes.namespace`.

Runtime behaviour:

| Invocation | What happens |
|---|---|
| `--target k8s` | preflight (cross-arch: builder `docker info` arch vs. `kubectl get nodes -L kubernetes.io/arch`) → `bentoml build --version <tag>` (default tag = short git SHA; bento version == git SHA == image tag) → topology from the fresh bento's `bento.yaml`, refreshing `deploy/.bento-topology.json` → `bentoml containerize -t <image>:<tag>` (`--opt platform=linux/<node arch>` if archs differ) → push (ECR login + describe-or-create if the URL host is ECR) → render in memory: one Deployment + Service per BentoML service, an HPA per autoscaled service, optional Ingress → `kubectl apply -f -` → `rollout status` for **every** Deployment in derived order (topological sort of the bento's DAG, deepest deps first) → `/readyz` + optional inference smoke test on the **entry** service via port-forward → the dependency-call proof. Single-service bento = one Deployment + Service. |
| `--target k8s --render-only [DIR]` | Manifests to disk (default `deploy/rendered/`), exit 0, **no cluster contact**: review-before-apply, config diffs, GitOps. |
| `--target ec2` | Same preflight/build/push; then per host over SSH: optional ECR token-over-stdin login → `docker pull` → `rm -f` + `run -d --restart unless-stopped -p <host_port>:3000` (secret env values via stdin, never a command line) → container-Up gate → `/readyz` + inference smoke test over a hardened SSH tunnel or direct HTTP. |
| Flags | `--check-only` (+ `--local-only` for credential-free CI PR gates), `--render-only [DIR]`, `--skip-build` (preflights that the image exists), `--image REF`, `--version TAG`, `--no-verify`, `--config`, `--output-json`. Exit codes 0..7; JSON summary as the last stdout line (contract in the bundle's README). |
| Requirement | **PyYAML** — config.yml is YAML, stdlib has no parser. `bentoml` depends on it; deploy-only environments need `pip install pyyaml`. Preflight name: `common.pyyaml`. |

## Step 0 — Locate the project and confirm scope

Find the BentoML project (`service.py` / `bentofile.yaml`; detection as in
`bentoml-containerize`). The bundle goes into `<project>/deploy/`; if that exists,
show its contents and get explicit confirmation before overwriting. Both **k8s**
and **ec2** produce working scripts. Prerequisites the script cannot create:

| Prerequisite | Detail |
|---|---|
| Existing EC2 instances | ec2 deploys to **existing instances only**. Need a new one? Provision it with the interactive `bentoml-ec2-deploy` skill first, then generate this bundle against the resulting host(s). |
| An existing namespace | Never creates namespaces, never renders a Namespace object; preflight fails with the exact `kubectl create namespace` line. |
| A registry | **Writable, and the cluster can pull from it** — or, for kind/minikube, nodes you can `image load` onto. |
| `.gitignore` entries | `__pycache__/`, `deploy/rendered/`, `deploy/.bento-topology.json` — all generated. `bentoml build` writes `__pycache__/` inside the project; an uncovered artifact makes every later run warn about a dirty git tree. The topology cache may be committed deliberately (Step 2). |

An older bundle (`deploy/deploy.config.json` + hand-owned manifests in
`deploy/k8s/`, or a `config.yml` stamped `bentoml-deploy-config/v3`) means
**regeneration, not migration**: carry the settings over, drop everything v4
derives. Pre-v4 configs are rejected with exit 2 and a message naming what moved.
Old hand-edited manifests still work via `kubernetes.manifests_dir`; rendering is
the default and the recommendation.

| Old key | Now |
|---|---|
| `project.dir` | `project` (a scalar) |
| `project.service` | the optional top-level `build_target` |
| `project.name` | gone; the bento's own name feeds the `app.kubernetes.io/part-of` label |
| `image.registry` + `image.repository` | one `image:` URL with **no tag** |
| `image.registry_type`, `image.ecr_region` | derived from the URL host |
| `image.platform` | auto-detected |
| `image.estimated_size_gb`, `image.local_image_preloaded` | gone |
| `services.<Name>.entry`, `.depends` | **deleted**; topology comes from `bento.yaml`, and `services:` is optional and overrides-only |

## Step 1 — Gather parameters (one round of questions)

Detect what you can, then ask the rest in one round; conventions as in the
interactive deploy skills. Read `service.py` for the entry service class, its
`@bentoml.api` methods (the smoke test) and any `@bentoml.service(resources=...)`
worth mirroring; run `kubectl config get-contexts` — **never assume the current
context**. Never ask about topology: no service list, entry flag, `depends`,
rollout order, dependency URL or slug goes into the config.

Only four values are REQUIRED:

| Parameter | Config key | Default / notes |
|---|---|---|
| Project root | `project` | `..` for the bundle layout, `.` for a root-level config. Resolved against the config file's directory; must exist. |
| Image repository | `image` | `<registry>/<repository>`, **NO tag** — e.g. `123456789012.dkr.ecr.us-west-1.amazonaws.com/text-suite`, `ghcr.io/acme/text-suite`. ECR recognized from the host (login + describe-or-create automated); other hosts: the user keeps `docker login` valid. `""` for kind/minikube local-load — nothing pushed, image named after the bento. |
| kubectl context | `kubernetes.context` | From `kubectl config get-contexts`; user confirms explicitly. |
| Namespace | `kubernetes.namespace` | Must already exist (preflight checks). Embedded in every derived dependency URL. |

Everything else is an override with a working default — ask only about what the
user wants to change:

| Parameter | Config key | Default / notes |
|---|---|---|
| Build target | `build_target` | `module:Class`, e.g. `service:TextPipeline`. Omit unless plain `bentoml build` cannot resolve the service. |
| Replicas / CPU / memory / GPU | `services.<Name>.replicas`, `.resources` | 1 replica; requests `cpu 500m`/`memory 1Gi`; limits `cpu "2"`/`memory "4Gi"`. **Quantities are quoted strings** (`cpu: "2"`); unquoted numbers are rejected. Mirror `@bentoml.service(resources=...)` (inert in OSS BentoML) to make those values bind. A block REPLACES the default wholesale. |
| Exposure | `services.<entry>.expose` | ClusterIP by default, verified via port-forward. `NodePort`/`LoadBalancer` and `node_port` (30000–32767, NodePort only) live here — **entry service only**; `expose`/`ingress` elsewhere are rejected. Key by `bento.yaml`'s `entry_service` name. |
| Ingress | `services.<entry>.ingress` | Optional; needs a cluster controller. `host` required when enabled. |
| Autoscaling | `services.<Name>.autoscaling` | `enabled: true` renders an HPA **and** drops that Deployment's `replicas`. `metric: cpu` requires `resources.requests.cpu` (enforced); `metric: concurrency` needs prometheus-adapter/KEDA publishing the `bentoml_inflight` Pods metric. |
| Runtime retuning | `services.<Name>.config_overrides` | **Bare** per-service keys (`{workers: 2, traffic: {timeout: 120}}`); the renderer nests them under `{"services": {"<Name>": …}}` for `BENTOML_CONFIG_OVERRIDES`. See bentoml-k8s-deploy references/customization.md. |
| Env / secrets (k8s) | `services.<Name>.env`, `.env_from_secrets` | Plain quoted-string values in `env`; secrets by Secret name in `env_from_secrets` (Secret must already exist). |
| Placement | `services.<Name>.node_selector`, `.tolerations` | GPU pools, arch, zones. Also the answer for a mixed-architecture cluster. |
| Probes | `services.<Name>.probes` | `startup_failure_threshold` (x 10 s = model-load budget, default 60); `readiness_timeout_seconds` (**floor of 6**, enforced — BentoML gives a dependency a hard-coded 5 s budget when `/readyz` fans out). |
| Slug | `services.<Name>.slug` | Derived from the service name (snake_case, `_`→`-`); set only when the derived value is not a valid DNS-1035 label (starts with a digit) or collides. |
| Extra labels | `kubernetes.extra_labels` | Team/cost-center labels merged onto every rendered object. |
| Pull secret | `kubernetes.image_pull_secret` | Private-registry Secret name, when the cluster cannot pull natively (EKS→ECR usually can). Existence is checked; an ECR-token secret older than 11 h warns (heuristic). Creating/refreshing: manual/CI. |
| Dependency-call proof | `verify.dependency_metrics` (default `true`) | Multi-service + inference block only: samples each dependency's `bentoml_service_request_total` around the request, failing if it did not move — the only catch for a dependency BentoML instantiated **in-process** (green `/readyz`, correct answer, idle pods). Leave `true`; without `verify.inference` it logs as skipped, not disabled. |
| Inference smoke test | `verify.inference.path` / `.body` / `.expect_substring` | **Optional.** From an `@bentoml.api` method **on the entry service** (the only one verify talks to): path `/<method>`, native-YAML body for its params, and a substring only a correct answer can contain — never one the service echoes back. Omit for `/readyz`-only verification; say what that gives up. |

No platform or registry-type question — both derived. Arch mismatches (the
`exec format error` class) are detected and cross-built automatically; buildx
presence is preflighted.

### Which config file to write

Both templates ship; make exactly one of them `deploy/config.yml`, and say which
and why:

- **`config.minimal.yml`** — every default accepted, nothing beyond the four
  required values. Four lines, no comments. Say plainly it **cannot** express a
  NodePort, Ingress, autoscaling, per-service resources or a smoke test:
  all-defaults means ClusterIP verified through a port-forward.
- **`config.yml`** (annotated) — anything else customized, mirrored decorator
  resources and NodePort/smoke-test requests included; where most multi-service
  bentos land. Prune inapplicable blocks and `ec2:`/`kubernetes:` sections, keep
  the comments.

### EC2 target parameters

Same round as `bentoml-ec2-deploy`'s Step 0, minus provisioning; all keys sit in
the **top-level `ec2:`** section. Arch needs no question: preflight compares each
host's `uname -m` with the builder's, cross-builds on a mismatch, and fails before
anything mutates if two hosts disagree.

| Parameter | Config key / placeholder | Default / notes |
|---|---|---|
| Hosts | `{{EC2_HOST}}` → `ec2.hosts` | Public IPs/DNS names of **existing** instances (YAML list, 1..N; bare hosts, no `user@`). Ask for all now — the script loops per host, fail-fast — and they must share ONE architecture (one image for all). |
| SSH user | `{{EC2_SSH_USER}}` → `ec2.ssh_user` | `ec2-user` (Amazon Linux) or `ubuntu` (Ubuntu). |
| SSH key path | `{{EC2_SSH_KEY_PATH}}` → `ec2.ssh_key_path` | Private key path (`~` expands; relative paths resolve against the config file's directory). Point at the key's usual home (`~/.ssh/...`), never a copy in the repo. Preflight enforces mode 600. |
| Container name | `{{CONTAINER_NAME}}` → `ec2.container_name` | The container `docker rm -f` + `run` replaces every deploy. Defaults to the project directory name; set it to the bento name with `_`→`-`. |
| Host port | `ec2.host_port` (ships `3000`) | Published as `-p <host_port>:3000`. Preflight fails if anything but our container holds it. |
| Runtime env var names | `ec2.env_names` (ships `[]`) | NAMES only, e.g. `["HF_TOKEN"]`; values come from the deploying shell/CI environment at run time, never written anywhere. |
| Registry auth | `{{EC2_REGISTRY_AUTH}}` → `ec2.registry_auth` | `"ecr-token-over-ssh"` for ECR when the deploying machine/CI holds AWS credentials: a fresh token piped to each host per run, the only method that never touches instance IAM. `"preauthed"` when instances pull themselves — instance profile, a user-maintained docker login, or a public image. |
| Verify path | `ec2.verify_via` (ships `"tunnel"`) | `"tunnel"` needs no inbound rule for the host port; `"direct"` only when the security group allows `host_port` from the deploying machine — ask, never assume. |
| Local tunnel port | `ec2.local_tunnel_port` (ships `3230`) | Tunnel verify only; change it if 3230 is taken locally. |

## Step 2 — Copy the bundle VERBATIM and render the config

```bash
mkdir -p <project>/deploy
cp -R <this-skill>/templates/deploy/. <project>/deploy/
# MANDATORY: drop bytecode caches. Running the templates in place (e2e tests, a stray
# `python3 -m py_compile`) leaves __pycache__/ in the skill; `cp -R` would copy it into
# the user's repo as stale, committable dirt.
find <project>/deploy -name __pycache__ -type d -prune -exec rm -rf {} +
find <project>/deploy -name '*.pyc' -delete
# Verify the copy before rendering; caches on EITHER side would show up as spurious
# differences, hence the exclude. Must report NO differences at all.
diff -r --exclude=__pycache__ <this-skill>/templates/deploy <project>/deploy
# Pick ONE config template, delete the other. EITHER all defaults accepted, the 4-line
# file overwriting the annotated one:
mv <project>/deploy/config.minimal.yml <project>/deploy/config.yml
# OR anything customized, keeping the annotated config.yml:
rm <project>/deploy/config.minimal.yml
```

Afterwards the only differences from the templates may be `config.yml`, the removed
config template, and `README.md`. Every `.py` file stays byte-identical forever, and
no `k8s/` dir is ever added.

Render placeholders in **exactly two files** — `deploy/config.yml` and
`deploy/README.md` — file by file. **Never a blanket sed across the bundle**: the
`.py` files must stay byte-identical, and some carry literal
`{{SERVICE_NAME}}`-style text in comments a global sed would corrupt.

- Both config templates: `{{IMAGE_URL}}`, `{{K8S_CONTEXT}}`, `{{NAMESPACE}}`.
- Annotated only: `{{ENTRY_SERVICE_NAME}}`, `{{DEP_SERVICE_NAME}}`,
  `{{INFERENCE_PATH}}`, `{{INFERENCE_BODY_TEXT}}`, `{{EXPECT_SUBSTRING}}`,
  `{{EC2_*}}`, `{{CONTAINER_NAME}}`.
- `README.md`: the same three, plus `{{SERVICE_NAME}}` (its title — the bento
  name) and `{{ENTRY_SERVICE_SLUG}}` / `{{DEP_SERVICE_SLUG}}` for the rollback
  recipe (derived slugs: service names snake_cased, `_`→`-`). The literal
  `{{PLACEHOLDER}}` in its preflight prose is prose — leave it.

Rules for the two rendered files:

| Topic | Rule |
|---|---|
| Comments | **Keep the annotated config's comments** — the only file the user edits; they carry the BentoML reasoning (readiness floor, pickle/RCE boundary, in-process-dependency trap, HPA-vs-replicas, config_overrides nesting). Prune only blocks you delete outright; never strip comments. Minimal file: **comment-free**. |
| Quoting | Placeholders sit inside quotes (`context: "{{K8S_CONTEXT}}"`) — substitute *inside* them. An unquoted `{{...}}` is invalid YAML (`{` starts a flow mapping): keep the quotes when rewriting a line, or drop them only for a known-safe plain scalar. |
| `image:` | **NO tag** (rejected by the loader): `<registry>/<repository>` only; the tag is the bento version. `""` only for kind/minikube. |
| `services:` | Optional, overrides-only, keyed by BentoML service name exactly as the bento spells it. Rename `"{{ENTRY_SERVICE_NAME}}"` to the entry service (the only block that may carry `expose:` / `ingress:`), keep or delete the commented `"{{DEP_SERVICE_NAME}}"` example, delete the section when nothing needs an override. Never write `entry:`/`depends:` (loader rejects both), a rollout order, a dependency URL, `BENTOML_SERVE_DEPENDS`, or `BENTOML_RUNNER_MAP`. |
| Unused sections | Delete `ec2:` for a k8s-only bundle, or `kubernetes:` **and** `services:` for an ec2-only one — a clean block deletion in YAML. |
| `verify.inference` | Optional: replace `body:` with the real request body (any JSON-able shape, native YAML), pick an `expect_substring` only a correct answer can contain; or delete the `inference:` block for `/readyz`-only verification — say what that gives up (multi-service: the dependency-call proof logs as skipped). |
| Optional keys | Ship as `null` (`build_target`, `kubernetes.image_pull_secret`, `kubernetes.manifests_dir`, `expose.node_port`, `ingress.tls_secret`); replace only when the parameter applies. |
| `schema` | OPTIONAL — absent means the current version, hence no such key in the minimal file. Keep `schema: bentoml-deploy-config/v4` in the annotated one. |
| `manifests_dir` | **Leave `null`** unless the user needs Kubernetes fields the schema lacks (sidecars, volumes, PDBs, affinity, `terminationGracePeriodSeconds`, `imagePullPolicy`, HPA `behavior`, probe paths for a `path_prefix` service). Then: render once with `--render-only DIR`, let them edit, point the key at `DIR` — and warn that those files are applied **verbatim, image ref included**, so every build means re-rendering or retagging, and they own the wiring. |
| Topology cache | Tell the user to **commit `deploy/.bento-topology.json` rather than ignore it** if their CI must render or run `--check-only --local-only` without docker and without `bentoml` installed: it is the third topology source. Re-commit whenever the bento's service set changes. |
| README — prune | Drop chapters, CI/CD jobs and rollback sections for targets you did not generate. **Retarget the generic examples** — Usage lines, CI one-liners, sample JSON summary and the GitLab CI deploy job are k8s-flavored (`--target k8s`, `"target": "k8s"`). For ec2-only, rewrite them to ec2 (per the chapter's own notes: SSH key from a CI secret file-variable, no kubectl/kubeconfig, no dind when `--skip-build`) and drop the rendering/`--render-only`/`manifests_dir` material (k8s-only). Prune target-*specific* prose only: chapters, CI jobs, rollback recipes, secrets-wiring rows. |
| README — keep | **Keep shared reference material even where it names the other target**: the `--target {k8s,ec2}` flag row, the `BENTOML_DEPLOY_EC2_HOSTS` override, the stage-naming section listing `k8s.apply`, `k8s.rollout[<slug>]`, `ec2.deploy[<host>]`. They document the unchanging contract of the one `deploy.py` in every bundle; residual mentions of the other target there are correct. |

Validate: `python3 -c "import yaml,sys;yaml.safe_load(open('deploy/config.yml'))"`
parses; `grep -nE '\{\{[A-Z][A-Z0-9_]*\}\}' deploy/config.yml deploy/README.md`
finds nothing (it matches generator placeholders only, not the `${{ secrets.* }}`
expressions in the README's GitHub Actions examples);
`find deploy -name __pycache__ -o -name '*.pyc'` is empty.

## Step 3 — Render the manifests and review them (k8s target only)

Never hand-write manifests, and never copy `bentoml-k8s-deploy`'s templates in:
`deploy/_internal/render.py` produces those shapes from `config.yml`. Skip this
step for an ec2-only bundle.

```bash
# --render-only needs two things. (1) An image ref for the Deployments: --version (with
# `image:`), --image REF, or a git checkout (default tag = short git SHA); nothing is
# built, the ref is only written into the YAML. (2) The topology, from the image (docker
# pulls/reads it) or from deploy/.bento-topology.json; if the image does not exist yet,
# build the bento once (`bentoml build`) or run a full deploy first -- the failure
# message names all three sources.
python3 deploy/deploy.py --target k8s --render-only --version review
kubectl --context <ctx> apply --dry-run=client -f deploy/rendered   # optional but cheap
```

Confirm what only the user knows: resources per service, exposure. The topology is
not theirs to confirm — it came from the bento — but show it: `--render-only` prints
the derived rollout order, and the log line names which source answered.
`BENTOML_SERVE_DEPENDS` on non-leaf services is the renderer's property, not
anything the user wrote.

- The deploy path renders in memory and never reads `deploy/rendered/`; delete it
  after the review. Re-rendering prunes stale files it owns (renamed slug, deleted
  service, HPA off) and warns about files it does not.
- The render also (re)writes `deploy/.bento-topology.json` whenever it discovers
  the topology freshly.
- Rendered objects carry `app.kubernetes.io/managed-by: bentoml-k8s-deploy`
  whichever path applied them, so interactive-then-CI cannot drift and no label is
  ever rewritten.

## Step 4 — Prove the bundle works, then hand it over

**Commit the bundle first** (`git add deploy/ && git commit`): the image tag
defaults to the project's short git SHA, so an uncommitted `deploy/` makes the gate
warn `working tree is dirty — the git-SHA image tag will not uniquely identify
this build`, and the build would carry a stale SHA. To review first, run the gate
anyway and say the warning is expected until they commit. Run it as CI would, once
per generated target:

```bash
python3 deploy/deploy.py --target k8s --check-only
python3 deploy/deploy.py --target ec2 --check-only        # connects to every host over SSH
```

The full ec2 `--check-only` needs the SSH key, per-host reachability, any
`env_names` values exported, and (with `ecr-token-over-ssh`) AWS credentials; it
probes each host's docker daemon, arch and host port in one SSH round trip per
host, changing nothing.

Show the check list and the summary JSON; it must exit 0 before you hand off.
Lacking credentials, degrade explicitly and name what was skipped (the summary's
`skipped_checks` lists it):

- no bentoml CLI / registry credentials, cluster access works:
  `python3 deploy/deploy.py --target k8s --check-only --skip-build --image <known-ref>`
- no cluster/host/AWS or registry access at all (CI PR gate, or ec2 without AWS
  credentials here):
  `python3 deploy/deploy.py --target <k8s|ec2> --check-only --local-only`

If a check fails, fix the *environment or config* it names — never the scripts.
Then tell the user:

| Topic | What to say |
|---|---|
| Generated files | `deploy/config.yml` (**the only file they edit**), `deploy/deploy.py` + `deploy/_internal/` (never edit), `deploy/README.md`. No manifests; `--render-only` shows the rendered objects. |
| How the k8s run behaves | Runtime sequence above. Entry rolls last: a caller's `/readyz` fans out to its dependencies. Verification goes through the **entry** Service only. Per-service `k8s.rollout[<slug>]` stages in the JSON summary; `k8s.apply` records the YAML's origin. The dependency request-counter proof catches a dependency BentoML ran in-process — otherwise the run looks perfectly healthy. |
| How to change it | Edit `config.yml`, optionally `--render-only` to review the diff, re-run. Adding/removing a BentoML service needs **no config change**: declare it in `service.py` with its `bentoml.depends(...)`, re-run, and it gets a Deployment, a Service, its rollout position and its dependency URLs. Only non-default resources or exposure need a `services:` block. |
| How to deploy | `python3 deploy/deploy.py --target <k8s\|ec2>` (build+push+deploy+verify); `--skip-build --image REF` for redeploys/rollbacks. For ec2, export the secrets named in `env_names` first. Rolling back an image is not rolling back the config: `config.yml` at the current checkout is what gets rendered. |
| Commit it | No secrets in the bundle — `git add deploy/ && git commit -m "Add production deploy bundle"`. |
| Wire CI later | The CI/CD chapter in the generated `deploy/README.md`: full GitHub Actions workflow with AWS OIDC and per-target deploy jobs, optional render-diff job, GitLab CI equivalent, `--check-only --local-only` as the fork-safe PR gate. A job that does not `pip install bentoml` still needs `pip install pyyaml`. |
| When a deploy fails | The runbook is `bentoml-k8s-deploy/references/troubleshooting.md`. |
