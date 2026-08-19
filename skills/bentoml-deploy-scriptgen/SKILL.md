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

**This bundle is also the interactive path's renderer.** `bentoml-k8s-deploy`
copies `templates/deploy/` into the project with the very commands in Step 2
below and calls `deploy.py --target k8s --render-only`, so
`_internal/render.py` + `_internal/config.py` are the single source of truth for
both flows — there is no second renderer to keep in sync. Two config layouts are
supported and both are exercised:

- **bundle layout** (this skill): `deploy/config.yml` with `project: ..`;
  `python3 deploy/deploy.py --target k8s [--render-only]` finds it next to
  `deploy.py`.
- **root-level layout**: `config.yml` at the project root with
  `project: .`; pass it explicitly —
  `python3 deploy/deploy.py --target k8s --render-only k8s --config config.yml`.
  Every relative path in the config (`project`, `kubernetes.manifests_dir`)
  resolves against **the config file's own directory**, so both layouts behave
  identically.

**config.yml v4 is overrides-only.** The service list, the entry service and the
dependency DAG are read from the bento's own `bento.yaml` at RUN time (local
bento store after the build, `docker run --rm --entrypoint cat <image>
$BENTO_PATH/bento.yaml` under `--skip-build`, or the `deploy/.bento-topology.json`
cache) — so you never write a service list, an `entry:` flag or a `depends:` list
into the config, and there is nothing to keep in sync with the code. The image
tag is the bento version, ECR is recognized from the image URL's host, and a
cross-architecture build is auto-detected. **Four values are enough to deploy**:
`project`, `image`, `kubernetes.context`, `kubernetes.namespace`.

You are a **script generator, not a script author**. The Python files under
this skill's `templates/deploy/` are static, reviewed, e2e-tested artifacts.

**HARD RULE — copy the `.py` files VERBATIM. Never edit, patch, "improve",
or regenerate their code, and never write new script logic for the user.**
Everything user-specific flows through exactly **two** rendered files:
`config.yml` and `README.md`. There are no manifests to write: the bundle
**renders** the Kubernetes manifests from `config.yml` on every run. If a user
need cannot be expressed in `config.yml`, say so, point at
`kubernetes.manifests_dir` (the documented escape hatch), and report it as a
limitation — do not fork the templates.

What the generated bundle does at runtime (driven by `config.yml` + the bento):

- `python3 deploy/deploy.py --target k8s` → preflight (incl. cross-arch
  detection: builder `docker info` arch vs. `kubectl get nodes -L
  kubernetes.io/arch`) → `bentoml build --version <tag>` (tag defaults to the
  project's short git SHA, so bento version == git SHA == image tag) → read the
  fresh bento's `bento.yaml` for the topology (and refresh
  `deploy/.bento-topology.json`) → `bentoml containerize -t <image>:<tag>`
  (`--opt platform=linux/<node arch>` when the archs differ) → push (ECR login +
  describe-or-create handled automatically when the URL host is ECR) →
  **render** one Deployment + Service per BentoML service (plus an HPA per
  autoscaled service and an optional Ingress) in memory and pipe them to
  `kubectl apply -f -` → `rollout status` for **every** Deployment in the
  **derived** rollout order (topological sort of the bento's DAG, deepest
  dependencies first) → `/readyz` + the optional inference smoke test against
  the **entry** service through a port-forward, then proof that every dependency
  really served a request. A single-service bento degenerates to exactly one
  Deployment + Service.
- `python3 deploy/deploy.py --target k8s --render-only [DIR]` → writes the
  rendered manifests (default `deploy/rendered/`) and exits 0 **without
  touching the cluster**: the review-before-apply, diff-a-config-change and
  GitOps path.
- `python3 deploy/deploy.py --target ec2` → same preflight/build/push, then
  per host over SSH: optional ECR token-over-stdin login → `docker pull` →
  `rm -f` + `run -d --restart unless-stopped -p <host_port>:3000` (secret
  env values travel via stdin, never in a command line) → container-Up gate
  → `/readyz` + inference smoke test through a hardened SSH tunnel (or
  direct HTTP). **Existing instances only — the script never provisions
  EC2.**
- Flags: `--check-only` (with `--local-only` for credential-free CI PR
  gates), `--render-only [DIR]`, `--skip-build` (which preflights that the
  image actually exists), `--image REF`, `--version TAG`, `--no-verify`,
  `--config`, `--output-json`. Exit codes 0..7 and a JSON summary as the last
  stdout line (contract documented in the bundle's README).

The bundle needs **PyYAML** (config.yml is YAML and the stdlib has no parser).
`bentoml` already depends on it; a deploy-only environment needs
`pip install pyyaml`. Preflight reports it as `common.pyyaml`.

## Step 0 — Locate the project and confirm scope

Find the BentoML project (directory containing `service.py` /
`bentofile.yaml`, same detection as the `bentoml-containerize` skill). The
bundle is generated into `<project>/deploy/`. If `deploy/` already exists,
show what is there and get explicit confirmation before overwriting.

Scope: the **k8s** and **ec2** targets both generate working scripts.
Standing prerequisites the script cannot create for the user:

- ec2 deploys to **existing instances only**; if an instance must be
  created, provision it first with the interactive `bentoml-ec2-deploy`
  skill, then generate this bundle against the resulting host(s).
- The k8s namespace must already exist (the bundle never creates namespaces
  and never renders a Namespace object; preflight fails with the exact
  `kubectl create namespace` line).
- A **writable image registry the cluster can pull from** (or, for the
  kind/minikube case, a cluster whose nodes you can `image load` onto).
- Make sure the project's `.gitignore` covers `__pycache__/`,
  `deploy/rendered/` and `deploy/.bento-topology.json` (all generated output):
  `bentoml build` creates the first inside the project, and an uncovered
  artifact makes every later run warn about a dirty git tree. The topology cache
  is the one file a user may want to commit deliberately — see Step 2.

If the project already has an older bundle (`deploy/deploy.config.json` with
hand-owned manifests under `deploy/k8s/`, or a `config.yml` stamped
`bentoml-deploy-config/v3`), this is a **regeneration**, not a migration: carry
the old settings into the new `config.yml` yourself and drop everything v4
derives. The bundle refuses a pre-v4 config with exit 2 and a message naming
what moved:

- `project.dir` → `project` (a scalar); `project.service` → the optional
  top-level `build_target`; `project.name` → gone (the bento's own name is used
  for the `app.kubernetes.io/part-of` label);
- `image.registry` + `image.repository` → one `image:` URL with **no tag**;
  `image.registry_type`, `image.ecr_region` → derived from the URL host;
  `image.platform` → auto-detected; `image.estimated_size_gb`,
  `image.local_image_preloaded` → gone;
- `services.<Name>.entry` and `.depends` → **deleted**; the topology comes from
  `bento.yaml`. `services:` is now optional, overrides-only.

Their old manifests are still usable as-is via `kubernetes.manifests_dir` if they
had hand-edited them — but the default and the recommendation is rendering.

Also make sure `.gitignore` covers `deploy/.bento-topology.json` (the topology
cache the bundle writes) unless the user's CI needs it committed — see Step 2.

## Step 1 — Gather parameters (one round of questions)

Same conventions as the interactive deploy skills. Detect what you can
(read `service.py` for the entry service class and its `@bentoml.api` methods;
run `kubectl config get-contexts` for the context list — **never assume the
current context**), then ask for the rest in one go.

**You do NOT need the bento's service topology.** No service list, no entry
flag, no `depends`, no rollout order, no dependency URL, and no slug goes into
the config: the bundle reads all of it from `bento.yaml` at run time. Read
`service.py` only to find the entry service's API routes for the smoke test, and
(optionally) the `@bentoml.service(resources=...)` values worth mirroring.

Only four values are REQUIRED:

| Parameter | Config key | Default / notes |
|---|---|---|
| Project root | `project` | `..` for this bundle (`deploy/config.yml`), `.` for a root-level config. Resolved against the config file's own directory; must exist. |
| Image repository | `image` | `<registry>/<repository>` with **NO tag**, e.g. `123456789012.dkr.ecr.us-west-1.amazonaws.com/text-suite`, `ghcr.io/acme/text-suite`. ECR is recognized from the host (login + describe-or-create automated); any other host means the user must keep `docker login` valid. `""` for the kind/minikube local-load case — then nothing is pushed and the image is named after the bento. A writable registry the cluster can pull from is a prerequisite. |
| kubectl context | `kubernetes.context` | From `kubectl config get-contexts`; the user must confirm it explicitly. |
| Namespace | `kubernetes.namespace` | Must already exist (preflight checks it). It is embedded in every derived dependency URL. |

Everything else is an override with a working default — ask only about what the
user actually wants to change:

| Parameter | Config key | Default / notes |
|---|---|---|
| Build target | `build_target` | `module:Class`, e.g. `service:TextPipeline`. Omit entirely unless plain `bentoml build` cannot resolve which service to build. |
| Replicas / CPU / memory / GPU | `services.<Name>.replicas`, `.resources` | Defaults: 1 replica, requests `cpu 500m`/`memory 1Gi`, limits `cpu "2"`/`memory "4Gi"`. **Every quantity is a quoted string** (`cpu: "2"`); the loader rejects unquoted numbers. Mirror the values declared in `@bentoml.service(resources=...)` (inert in OSS BentoML) when the user wants them to bind. A block you write REPLACES the default wholesale. |
| Exposure | `services.<entry>.expose` | Default ClusterIP, verified through a port-forward. `NodePort`/`LoadBalancer` and `node_port` (30000–32767, NodePort only) live here — **entry service only**; the loader rejects `expose`/`ingress` on any other service, and the entry service is `bento.yaml`'s `entry_service`, so use that name as the key. |
| Ingress | `services.<entry>.ingress` | Optional, entry service only; needs a controller in the cluster. `host` is required when enabled. |
| Autoscaling | `services.<Name>.autoscaling` | `enabled: true` renders an HPA **and** drops that Deployment's `replicas` automatically. `metric: cpu` requires `resources.requests.cpu` (enforced); `metric: concurrency` needs prometheus-adapter/KEDA publishing the `bentoml_inflight` Pods metric. |
| Runtime retuning | `services.<Name>.config_overrides` | **Bare** per-service keys (`{workers: 2, traffic: {timeout: 120}}`); the renderer nests them under `{"services": {"<Name>": …}}` for `BENTOML_CONFIG_OVERRIDES`. See bentoml-k8s-deploy's references/customization.md. |
| Env / secrets (k8s) | `services.<Name>.env`, `.env_from_secrets` | Plain values only in `env` (quoted strings); secrets by Secret name in `env_from_secrets` (the Secret must already exist). |
| Placement | `services.<Name>.node_selector`, `.tolerations` | GPU pools, arch, zones. Also the answer for a mixed-architecture cluster. |
| Probes | `services.<Name>.probes` | `startup_failure_threshold` (x 10 s = model-load budget, default 60) and `readiness_timeout_seconds` (**floor of 6**, enforced — BentoML gives a dependency a hard-coded 5 s budget when `/readyz` fans out). |
| Slug | `services.<Name>.slug` | Derived from the service name (snake_case, `_`→`-`); set it only when the derived value is not a valid DNS-1035 label (a name starting with a digit) or collides. |
| Extra labels | `kubernetes.extra_labels` | Team/cost-center labels merged onto every rendered object. |
| Pull secret | `kubernetes.image_pull_secret` | Secret name if the registry is private and the cluster cannot pull natively (EKS→ECR usually can). The script checks existence and warns when an ECR-token secret is older than 11 h (heuristic); creating/refreshing it stays a manual/CI step. |
| Dependency-call proof | `verify.dependency_metrics` (default `true`) | Multi-service bentos only, and only with an inference block: verify samples each dependency's `bentoml_service_request_total` around the request and fails if it did not move — the only way to catch a dependency BentoML silently instantiated **in-process** (green `/readyz`, correct answer, idle dependency pods). Leave it `true`. Without `verify.inference` the proof is logged as skipped, not disabled. |
| Inference smoke test | `verify.inference.path` / `.body` / `.expect_substring` | **Optional.** Derive from an `@bentoml.api` method **on the entry service** (the only one verify talks to): path `/<method>`, a native-YAML body for its params, and a substring only a correct answer can contain (never one the service echoes back from the request). Omit the block for `/readyz`-only verification, and say what that gives up. |

There is no platform question and no registry-type question: both are derived.
Arch mismatches (the `exec format error` class) are detected by comparing the
builder's architecture with the cluster's nodes (or each EC2 host's `uname -m`)
and cross-built automatically; buildx presence is preflighted.

### Which config file to write

Ship exactly one of the two templates as `deploy/config.yml`, and tell the user
which and why:

- **`config.minimal.yml`** — when the user accepted every default, i.e. they
  supplied nothing beyond `project`, `image`, `kubernetes.context` and
  `kubernetes.namespace`. Four lines, no comments.
- **`config.yml`** (annotated) — as soon as ANYTHING else is customized, which
  includes mirroring decorator resources and asking for a NodePort or a smoke
  test. Most multi-service bentos land here; that is intended. Prune the blocks
  and the `ec2:`/`kubernetes:` sections that do not apply, and keep the comments.

Note plainly that the minimal file **cannot** express a NodePort, an Ingress,
autoscaling, per-service resources or an inference smoke test — an all-defaults
deployment is ClusterIP verified through a port-forward.

### EC2 target parameters

Same question round as the interactive `bentoml-ec2-deploy` skill's Step 0,
minus provisioning (the script only ever deploys to existing instances). All
keys live in the **top-level `ec2:`** section:

| Parameter | Config key / placeholder | Default / notes |
|---|---|---|
| Hosts | `{{EC2_HOST}}` → `ec2.hosts` | YAML list of public IPs/DNS names of **existing** instances (1..N; bare hosts, no `user@`). Ask for all of them now — the script loops per host, fail-fast — and note they must all share ONE architecture (one image is built for all of them). |
| SSH user | `{{EC2_SSH_USER}}` → `ec2.ssh_user` | `ec2-user` (Amazon Linux) or `ubuntu` (Ubuntu). |
| SSH key path | `{{EC2_SSH_KEY_PATH}}` → `ec2.ssh_key_path` | Path to the private key (`~` expands; relative paths resolve against the config file's directory). Point at the key's usual home (e.g. `~/.ssh/...`) — never copy a private key into the repo. Preflight enforces mode 600. |
| Container name | `{{CONTAINER_NAME}}` → `ec2.container_name` | Names the container that `docker rm -f` + `run` replaces on every deploy. Defaults to the project directory's name; set it to the bento name with `_`→`-`. |
| Host port | `ec2.host_port` (ships `3000`) | Published as `-p <host_port>:3000`. Preflight fails if anything other than our container holds it. |
| Runtime env var names | `ec2.env_names` (ships `[]`) | NAMES only, e.g. `["HF_TOKEN"]` — values come from the deploying shell/CI environment at run time and are never written anywhere. |
| Registry auth | `{{EC2_REGISTRY_AUTH}}` → `ec2.registry_auth` | `"ecr-token-over-ssh"` for ECR images when the deploying machine/CI holds AWS credentials (a fresh token is piped to each host per run — the only method that never touches instance IAM). `"preauthed"` when the instances pull on their own: instance profile, a docker login the user maintains, or a public image. |
| Verify path | `ec2.verify_via` (ships `"tunnel"`) | `"tunnel"` (default; works with no inbound rule for the host port) or `"direct"` (only when the security group allows `host_port` from the deploying machine — ask, never assume). |
| Local tunnel port | `ec2.local_tunnel_port` (ships `3230`) | Only for tunnel verify; change it if 3230 is taken locally. |

Arch matching needs no extra question: preflight compares each host's
`uname -m` against the builder's architecture, cross-builds when they differ,
and fails before anything mutates if two hosts disagree.

## Step 2 — Copy the bundle VERBATIM and render the config

```bash
mkdir -p <project>/deploy
cp -R <this-skill>/templates/deploy/. <project>/deploy/
# MANDATORY: drop Python bytecode caches the copy may have dragged along.
# Running the templates in place (e2e tests, a stray `python3 -m py_compile`)
# leaves __pycache__/ dirs inside the skill, and `cp -R` copies them into the
# user's repo as stale, committable dirt.
find <project>/deploy -name __pycache__ -type d -prune -exec rm -rf {} +
find <project>/deploy -name '*.pyc' -delete
```

Verify the copy is verbatim before rendering anything (caches on **either**
side would otherwise show up as spurious differences, hence the exclude):

```bash
diff -r --exclude=__pycache__ <this-skill>/templates/deploy <project>/deploy
```

That must report **no differences at all** at this point. After rendering, the
only differences may be `config.yml`, the removal of the unused config template,
and `README.md` — every `.py` file stays byte-identical forever, and there is no
`k8s/` directory to add.

The bundle ships TWO config templates. **Pick one** (see "Which config file to
write" above), rename it to `config.yml`, and delete the other:

```bash
# all defaults accepted -> the 4-line file:
mv <project>/deploy/config.minimal.yml <project>/deploy/config.yml   # overwrites the annotated one
# anything customized -> keep the annotated one:
rm <project>/deploy/config.minimal.yml
```

Then render placeholders in **exactly two files**: `deploy/config.yml` and
`deploy/README.md`. Substitute only there, file by file — **never run a blanket
sed across the bundle**: the `.py` files must stay byte-identical to the
templates, and some contain literal `{{SERVICE_NAME}}`-style text in comments
that a global substitution would corrupt.

Placeholders in the two config templates: `{{IMAGE_URL}}`, `{{K8S_CONTEXT}}`,
`{{NAMESPACE}}` (both files), plus `{{ENTRY_SERVICE_NAME}}`,
`{{DEP_SERVICE_NAME}}`, `{{INFERENCE_PATH}}`, `{{INFERENCE_BODY_TEXT}}`,
`{{EXPECT_SUBSTRING}}` and the `{{EC2_*}}` / `{{CONTAINER_NAME}}` set in the
annotated one. `README.md` additionally has `{{SERVICE_NAME}}` (its title — use
the bento name), `{{ENTRY_SERVICE_SLUG}}` / `{{DEP_SERVICE_SLUG}}` (the rollback
recipe — the derived slugs, i.e. the service names snake_cased with `_`→`-`) and
the same `{{IMAGE_URL}}` / `{{K8S_CONTEXT}}` / `{{NAMESPACE}}` values. The
literal `{{PLACEHOLDER}}` in its preflight prose is prose, not a placeholder —
leave it.

YAML notes (much less booby-trapped than the old JSON config — no commas, and
`#` comments are real):

- **Keep the shipped comments** in the annotated file. It is the only file the
  user edits, and its comments carry the BentoML-specific reasoning (the
  readiness floor, the pickle/RCE boundary, the in-process-dependency trap, the
  HPA-vs-replicas rule, the config_overrides nesting). Prune only blocks you
  delete outright; never strip comments to "tidy up". Conversely, the minimal
  file must stay **comment-free** — that is its whole point.
- **Every placeholder is inside quotes on purpose** (`context: "{{K8S_CONTEXT}}"`).
  Substitute the value *inside* the quotes. An unquoted `{{...}}` is not valid
  YAML at all (`{` starts a flow mapping), so if you rewrite a line, keep the
  quotes — or drop them only for a value you know is a safe plain scalar.
- **`image:` carries NO tag** (the loader rejects one): `<registry>/<repository>`
  only. The tag is the bento version. Use `""` only for the kind/minikube case.
- **`services:` is optional and overrides-only**, keyed by BentoML service name
  exactly as the bento spells it. Rename `"{{ENTRY_SERVICE_NAME}}"` to the
  bento's entry service (that is the only block that may carry `expose:` /
  `ingress:`), keep or delete the commented `"{{DEP_SERVICE_NAME}}"` example, and
  **delete the whole `services:` section** when nothing needs an override. Never
  write `entry:` or `depends:` — the loader rejects both (they moved to
  `bento.yaml`) — and never write a rollout order, a dependency URL, a
  `BENTOML_SERVE_DEPENDS` value, or a runner-map env var.
- **Delete the whole `ec2:` section** for a k8s-only bundle, or the
  `kubernetes:` **and** `services:` sections for an ec2-only bundle. In YAML that
  is a clean block deletion — no comma surgery.
- `verify.inference` is optional: replace `body:` with the real request body (any
  JSON-able shape, native YAML) and pick an `expect_substring` only a correct
  answer can contain, or delete the whole `inference:` block for a `/readyz`-only
  verification (and say what that gives up: on a multi-service bento the
  dependency-call proof is then logged as skipped).
- Optional keys ship as `null` (`build_target`, `kubernetes.image_pull_secret`,
  `kubernetes.manifests_dir`, `expose.node_port`, `ingress.tls_secret`) —
  replace them only when the parameter applies.
- `schema` is OPTIONAL: absent means the current version, which is why the
  minimal file has no such key. Keep `schema: bentoml-deploy-config/v4` in the
  annotated file. A v1/v2 (`deploy.config.json`) or v3 config is rejected with
  exit 2 and a message naming what moved; there is no in-place migration.
- **Leave `kubernetes.manifests_dir: null`** unless the user explicitly needs
  Kubernetes fields the schema does not have (sidecars, volumes, PDBs, affinity,
  `terminationGracePeriodSeconds`, `imagePullPolicy`, HPA `behavior`, probe paths
  for a `path_prefix` service). Then, and only then: render once with
  `--render-only DIR`, let them edit those files, and point the key at `DIR` —
  and tell them those files are applied **verbatim, image ref included**, so
  every build means re-rendering or editing the tag, and they now own the
  wiring.
- **`.gitignore`**: `__pycache__/`, `deploy/rendered/` and
  `deploy/.bento-topology.json` (the topology cache the bundle writes next to the
  config). Tell the user to **commit that cache instead** if their CI has to
  render or `--check-only --local-only` without docker and without installing
  `bentoml` — that file is the third topology source, and re-commit it whenever
  the bento's service set changes.
- Prune the README the same way: drop the chapters, CI/CD jobs, and rollback
  sections for targets that were not generated, and **retarget the generic
  examples** to the target(s) you did generate: the Usage lines, the CI
  one-liners, the sample JSON summary, and the GitLab CI chapter's concrete
  deploy job in the template are all k8s-flavored (`--target k8s`,
  `"target": "k8s"`). For an ec2-only bundle, rewrite those to ec2 (adapt the
  GitLab job per the chapter's own notes: SSH key from a CI secret
  file-variable, no kubectl/kubeconfig, no dind needed when `--skip-build`) and
  drop the whole rendering/`--render-only`/`manifests_dir` material — it is
  k8s-only.
  **Where to stop:** prune only target-*specific* prose — chapters, CI jobs,
  rollback recipes, and the secrets-wiring rows for a target you did not
  generate. **Keep the shared reference material even when it names the other
  target**: the `--target {k8s,ec2}` flag row, the `BENTOML_DEPLOY_EC2_HOSTS`
  environment override, and the stage-naming section listing `k8s.apply`,
  `k8s.rollout[<slug>]` and `ec2.deploy[<host>]`. Those document the script's
  full, unchanging contract — the same `deploy.py` ships in every bundle — and
  editing them would misdescribe it. A few residual mentions of the other target
  in that reference material are correct, not leftovers.
- Validate before moving on:
  - `python3 -c "import yaml,sys;yaml.safe_load(open('deploy/config.yml'))"`
    (parses),
  - `grep -nE '\{\{[A-Z][A-Z0-9_]*\}\}' deploy/config.yml deploy/README.md`
    finds nothing (the pattern matches only generator placeholders, not the
    `${{ secrets.* }}` expressions that belong in the README's GitHub Actions
    examples),
  - `find deploy -name __pycache__ -o -name '*.pyc'` is empty.

## Step 3 — Render the manifests and review them with the user (k8s target only)

There are **no manifests to write**. Do not hand-write Kubernetes YAML, and do
not copy the `bentoml-k8s-deploy` templates into the bundle — the bundle's own
renderer (`deploy/_internal/render.py`) produces exactly those shapes from
`config.yml`. Your job is to prove it and to show the result:

```bash
# --render-only needs two things:
#   * an image ref for the Deployments: --version (with `image:`), --image REF,
#     or a git checkout (the default tag is the short git SHA). Nothing is
#     built; the ref is only written into the YAML.
#   * the bento's topology. Nothing is built here either, so it comes from the
#     image (docker pulls/reads it) or from deploy/.bento-topology.json. If the
#     image does not exist yet, build the bento once (`bentoml build`) or run
#     the full deploy first — the failure message names all three sources.
python3 deploy/deploy.py --target k8s --render-only --version review
kubectl --context <ctx> apply --dry-run=client -f deploy/rendered   # optional but cheap
```

Then read the output with the user and confirm the things only they know:
resources per service and exposure. The topology itself is NOT theirs to
confirm — it came from the bento — but do show it: `--render-only` prints the
derived rollout order, and the log line names which source answered.
`BENTOML_SERVE_DEPENDS` on every non-leaf service is now a property of the
renderer, not of anything the user wrote.

The ec2 target needs no manifests — skip this step entirely for an ec2-only
bundle.

Notes:

- `deploy/rendered/` is generated output. Either delete it after the review or
  add it to `.gitignore`; the deploy path renders in memory and never reads it.
  Re-rendering prunes stale files it owns (renamed slug, deleted service, HPA
  switched off) and warns about files it does not own.
- The render also (re)writes `deploy/.bento-topology.json` whenever it discovers
  the topology freshly. That is deliberate: it seeds the third source for later
  build-free runs.
- There is exactly ONE renderer and ONE validator: `_internal/render.py` and
  `_internal/config.py` in this bundle. The interactive `bentoml-k8s-deploy`
  skill does not implement either — it writes `config.yml` and then calls this
  bundle (`--render-only`, `--check-only`). So a project deployed interactively
  once and by CI forever after cannot drift: both paths run the same code on the
  same config. Rendered objects carry
  `app.kubernetes.io/managed-by: bentoml-k8s-deploy` regardless of which path
  applied them, so switching between them never rewrites a label.

## Step 4 — Prove the bundle works, then hand it over

**Commit the bundle first** (`git add deploy/ && git commit`). The image
tag defaults to the project's short git SHA, so running the gate on an
uncommitted `deploy/` warns `working tree is dirty — the git-SHA image tag
will not uniquely identify this build` and would tag a build with a stale
SHA. If the user prefers to review before committing, run the gate anyway
and tell them that warning is expected until they commit.

Run the preflight gate exactly as CI would, once per generated target:

```bash
python3 deploy/deploy.py --target k8s --check-only
python3 deploy/deploy.py --target ec2 --check-only        # connects to every host over SSH
```

For ec2, the full `--check-only` needs the SSH key, per-host reachability,
any `env_names` values exported, and (with `ecr-token-over-ssh`) AWS
credentials — it probes each host's docker daemon, architecture, and host
port in one SSH round trip per host, without changing anything. Without
those credentials on this machine, fall back to
`--check-only --local-only` and say so.

Show the user the check list and the summary JSON. It must exit 0 before
you hand off. If this machine lacks some credentials, degrade explicitly
and tell the user what was skipped (the summary's `skipped_checks` lists
it):

- no bentoml CLI / registry credentials, but cluster access works:
  `python3 deploy/deploy.py --target k8s --check-only --skip-build --image <known-ref>`
- no cluster/host/AWS or registry access at all (e.g. CI PR gate):
  `python3 deploy/deploy.py --target <k8s|ec2> --check-only --local-only`

If a check fails, fix the *environment or config* it names — never the
scripts.

Then tell the user:

1. **What was generated**: `deploy/config.yml` (**the only file they edit**),
   `deploy/deploy.py` and `deploy/_internal/` (never edit), `deploy/README.md`.
   No manifests: the Kubernetes objects are rendered from `config.yml` on every
   run, and `--render-only` shows them.
2. **How the k8s run behaves** (k8s target only): render → one
   `kubectl apply` → `rollout status` per Deployment in the derived order
   (deepest dependencies first, entry last, because a caller's `/readyz` fans
   out to its dependencies) → verification through the **entry** service's
   Service only. Each service gets its own `k8s.rollout[<slug>]` stage in the
   JSON summary, and `k8s.apply` records where the YAML came from. Verify then
   proves each dependency actually served a request (its own request counter
   must move); tell the user that this is what catches a dependency BentoML
   silently ran in-process, and that the run would otherwise look perfectly
   healthy.
3. **How to change the deployment**: edit `config.yml`, optionally
   `--render-only` to review the diff, re-run. Adding or removing a BentoML
   service needs **no config change at all** — declare it in `service.py` with
   its `bentoml.depends(...)`, re-run, and it gets a Deployment, a Service, its
   place in the rollout order and its dependency URLs. Only non-default
   resources or exposure need a `services:` block.
4. **How to deploy**: `python3 deploy/deploy.py --target <k8s|ec2>`
   (full build+push+deploy+verify), and the `--skip-build --image REF` form
   for redeploys/rollbacks. For ec2, secrets named in `env_names` must be
   exported in the deploying shell/CI environment first. Note that rolling back
   an image is not rolling back the config: `config.yml` at the current checkout
   is what gets rendered.
5. **Commit it**: the bundle contains no secrets —
   `git add deploy/ && git commit -m "Add production deploy bundle"` (and
   `.gitignore` `deploy/rendered/`).
6. **Wire CI later**: point them at the CI/CD chapter in the generated
   `deploy/README.md` (complete GitHub Actions workflow with AWS OIDC and
   per-target deploy jobs, an optional render-diff job for reviews, a GitLab CI
   equivalent, and `--check-only --local-only` as the fork-safe PR gate).
   Remind them that a job which does not `pip install bentoml` still needs
   `pip install pyyaml`.
