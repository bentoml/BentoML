---
name: bentoml-deploy-scriptgen
description: >
  Generate a standalone, committable production deploy-script bundle
  (deploy/deploy.py + deploy.config.json + rendered k8s manifests) that
  builds, containerizes, pushes, deploys, and verifies a BentoML service
  without any agent involved — runnable from a terminal or CI/CD. Use when
  the user says things like "generate a deployment script", "deploy from
  CI/CD", "set up a production deployment pipeline", "deploy without the
  agent", "give me a script I can commit to deploy this", or "automate my
  BentoML deploys". Complements the interactive skills: bentoml-containerize,
  bentoml-k8s-deploy, and bentoml-ec2-deploy do a one-off deploy with you in
  the loop; this skill emits scripts that repeat it forever. Kubernetes and
  EC2 targets.
---

# Generate a production deploy-script bundle

You are a **script generator, not a script author**. The Python files under
this skill's `templates/deploy/` are static, reviewed, e2e-tested artifacts.

**HARD RULE — copy the `.py` files VERBATIM. Never edit, patch, "improve",
or regenerate their code, and never write new script logic for the user.**
Everything user-specific flows through exactly two rendered files
(`deploy.config.json`, `README.md`) plus the k8s manifests. If a user need
cannot be expressed in `deploy.config.json`, say so and report it as a
limitation — do not fork the templates.

What the generated bundle does at runtime (all driven by config):

- `python3 deploy/deploy.py --target k8s` → preflight → `bentoml build
  --version <tag>` (tag defaults to the project's short git SHA) →
  `bentoml containerize -t <registry>/<repo>:<tag>` → push (ECR login +
  describe-or-create handled automatically) → `kubectl apply` of **all**
  manifests in `deploy/k8s/` in one pass with the image sentinel
  `__DEPLOY_IMAGE__` rewritten in memory → `rollout status` for **every**
  service's Deployment in `targets.k8s.services` order (dependencies first,
  entry last) → `/readyz` + inference smoke test against the **entry**
  service through a port-forward. One Kubernetes Deployment + Service per
  BentoML service; a single-service bento degenerates to exactly one of
  each.
- `python3 deploy/deploy.py --target ec2` → same preflight/build/push, then
  per host over SSH: optional ECR token-over-stdin login → `docker pull` →
  `rm -f` + `run -d --restart unless-stopped -p <host_port>:3000` (secret
  env values travel via stdin, never in a command line) → container-Up gate
  → `/readyz` + inference smoke test through a hardened SSH tunnel (or
  direct HTTP). **Existing instances only — the script never provisions
  EC2.**
- Flags: `--check-only` (with `--local-only` for credential-free CI PR
  gates), `--skip-build` (which preflights that the image actually exists),
  `--image REF`, `--version TAG`, `--no-verify`, `--config`,
  `--output-json`. Exit codes 0..7 and a JSON summary as the last stdout
  line (contract documented in the bundle's README).

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
- Make sure the project's `.gitignore` covers `__pycache__/` (and any other
  build artifacts): `bentoml build` creates it inside the project, and an
  uncovered artifact makes every later run warn about a dirty git tree.

## Step 1 — Gather parameters (one round of questions)

Same conventions as the interactive deploy skills. Detect what you can
(read `service.py` for the service class and `@bentoml.api` methods; run
`kubectl config get-contexts` for the context list — **never assume the
current context**), then ask for the rest in one go.

For the k8s target you also need the bento's **service topology** — the
service list, which one is the entry service, and the dependency order.
Get it the same way the `bentoml-k8s-deploy` skill does (its topology
discovery step: `bento.yaml`'s `entry_service` + `services`), and reuse the
slugs it derives, so the config and the manifests agree. Do not re-derive
the rules here.

| Parameter | Placeholder | Default / notes |
|---|---|---|
| Build target | `project.service` (ships `null`) | `module:Class`, e.g. `service:TextPipeline`. Leave `null` if `bentoml build` finds it alone; required when `service.py` defines several services. |
| Bento name | `{{SERVICE_NAME}}` | Bento name, `_`→`-`, must match DNS-1035 `^[a-z]([-a-z0-9]*[a-z0-9])?$` (same naming rule as bentoml-k8s-deploy). Names `project.name`, the ec2 container, and the README title — **not** the k8s objects (those are named per service, next row). |
| Services | `{{ENTRY_SERVICE_NAME}}` `{{ENTRY_SERVICE_SLUG}}` `{{DEP_SERVICE_NAME}}` `{{DEP_SERVICE_SLUG}}` → `targets.k8s.services` | One `{name, slug, entry}` object per BentoML service, **in rollout order: dependencies first, entry last**, exactly one `entry: true`. `name` is the BentoML service name as the bento declares it (whitespace-free); `slug` is that name snake_cased then `_`→`-`, DNS-1035 — it names `deployment/<slug>`, `svc/<slug>`, and the `<slug>-deployment.yaml` / `<slug>-service.yaml` files. Single-service bento → a **one-element** list with `entry: true`. |

**Worked example — a 4-service, 3-tier bento** (`Gateway` → {`Enricher`,
`Sentiment`}; `Enricher` → `Tokenizer`). The `{{DEP_SERVICE_*}}` placeholders are
**singular only because the template ships one example element** — duplicate the
object once per non-entry service, and keep the array in the rollout order the
`bentoml-k8s-deploy` skill derived (deepest tier first, then alphabetically):

```json
"services": [
  {"name": "Tokenizer", "slug": "tokenizer", "entry": false},
  {"name": "Sentiment", "slug": "sentiment", "entry": false},
  {"name": "Enricher",  "slug": "enricher",  "entry": false},
  {"name": "Gateway",   "slug": "gateway",   "entry": true}
]
```

Watch the JSON commas when you duplicate or delete elements (same hazard as
deleting a target block), and remember every non-leaf service — not just the
entry — needs its own `BENTOML_SERVE_DEPENDS` in its manifest; preflight's
`k8s.serve-depends` checks one pair per dependency edge, so a missing middle-tier
pair fails the gate.
| Registry host | `{{IMAGE_REGISTRY}}` | e.g. `123456789012.dkr.ecr.us-west-1.amazonaws.com`, `ghcr.io`, `docker.io`. `""` for local-only images. |
| Repository | `{{IMAGE_REPOSITORY}}` | e.g. `acme/text-pipeline`. Lowercase. |
| Registry type | `{{REGISTRY_TYPE}}` | `ecr` (auth + repo-create automated), `generic` (user must `docker login`), or `none` (kind/minikube local load — nothing pushed; the user must load the image and set `image.local_image_preloaded` to `true`, which preflight enforces). |
| Platform | `image.platform` (ships `null`) | Set `"linux/amd64"` when cluster nodes are amd64 and builds may run on arm64; leave `null` when archs match. |
| kubectl context | `{{K8S_CONTEXT}}` | From `kubectl config get-contexts`; user must confirm explicitly. |
| Namespace | `{{NAMESPACE}}` | Must exist (preflight checks it; the script never creates namespaces). |
| Replicas / CPU / memory / GPU | (manifest, **per service**) | Same defaults and per-service rules as bentoml-k8s-deploy — ask once per service; the values live only in the manifests. |
| Exposure | (manifest, **entry service only**) | ClusterIP (port-forward) / NodePort / LoadBalancer / Ingress — same rules as bentoml-k8s-deploy. Dependency Services are always ClusterIP: inter-service payloads are pickle and must never leave the cluster. |
| Autoscaling | (optional manifest) | Per bentoml-k8s-deploy. Any `<slug>-hpa.yaml` present in `deploy/k8s/` **is applied on every run** (the whole directory is applied) — generate one only if the user wants it, and tell them adding one later takes effect on the next deploy. **When you render an HPA for a service, that service's Deployment must NOT declare `spec.replicas`** (this bundle re-applies everything every run, so a declared `replicas:` resets the scale and fights the HPA). Preflight enforces it. |
| Dependency-call proof | `verify.dependency_metrics` (ships `true`) | Multi-service bentos only. Verify samples each dependency's `bentoml_service_request_total` around the inference request and fails if it did not move — the only way to catch a dependency that BentoML silently instantiated **in-process** (green `/readyz`, correct answer, idle dependency pods). Leave it `true`. Set `false` only if the smoke-tested API genuinely does not call some dependency; then say so to the user. |
| Fixed NodePort | `targets.k8s.node_port` (ships `null`) | Only for NodePort exposure when the user wants a stable URL: pick 30000–32767, set the config key AND render the same value as a `nodePort:` line in the **entry** service's `<slug>-service.yaml` (preflight cross-checks exactly that file). Leave `null` to let the cluster allocate (kubectl apply preserves an existing allocation). |
| Pull secret | `targets.k8s.image_pull_secret` (ships `null`) | Secret name if the registry is private and the cluster cannot pull natively (EKS→ECR usually can). The script checks existence and warns when an ECR-token secret is older than 11 h (heuristic — see the bundle README); creating/refreshing it stays a manual/CI step. |
| Inference smoke test | `{{INFERENCE_PATH}}` `{{INFERENCE_BODY}}` `{{EXPECT_SUBSTRING}}` | Derive from a `@bentoml.api` method **on the entry service** (that is the only one verify talks to): path `/<method>`, JSON body of its params, and a substring the response must contain (e.g. a result key). Set `"inference": null` only if the user declines. |

The k8s rows (context, namespace, services, replicas/resources, exposure,
autoscaling, NodePort, pull secret) apply only when generating the k8s
target; the manifest rows render into `deploy/k8s/`.

### EC2 target parameters

Same question round as the interactive `bentoml-ec2-deploy` skill's Step 0,
minus provisioning (the script only ever deploys to existing instances):

| Parameter | Placeholder / config key | Default / notes |
|---|---|---|
| Hosts | `{{EC2_HOST}}` → `targets.ec2.hosts` | JSON array of public IPs/DNS names of **existing** instances (1..N; bare hosts, no `user@`). Ask for all of them now — the script loops per host, fail-fast. |
| SSH user | `{{EC2_SSH_USER}}` → `targets.ec2.ssh_user` | `ec2-user` (Amazon Linux) or `ubuntu` (Ubuntu). |
| SSH key path | `{{EC2_SSH_KEY_PATH}}` → `targets.ec2.ssh_key_path` | Path to the private key (`~` expands; relative paths resolve against `deploy/`). Point at the key's usual home (e.g. `~/.ssh/...`) — never copy a private key into the repo. Preflight enforces mode 600. |
| Container name | `targets.ec2.container_name` (ships `{{SERVICE_NAME}}`) | Bento name with `_`→`-`. Names the container that `docker rm -f` + `run` replaces on every deploy. |
| Host port | `targets.ec2.host_port` (ships `3000`) | Published as `-p <host_port>:3000`. Preflight fails if anything other than our container holds it. |
| Runtime env var names | `targets.ec2.env_names` (ships `[]`) | NAMES only, e.g. `["HF_TOKEN"]` — values come from the deploying shell/CI environment at run time and are never written anywhere. |
| Registry auth | `{{EC2_REGISTRY_AUTH}}` → `targets.ec2.registry_auth` | `"ecr-token-over-ssh"` for ECR images when the deploying machine/CI holds AWS credentials (a fresh token is piped to each host per run — the only method that never touches instance IAM). `"preauthed"` when the instances pull on their own: instance profile, a docker login the user maintains, or a public image. |
| Verify path | `targets.ec2.verify_via` (ships `"tunnel"`) | `"tunnel"` (default; works with no inbound rule for the host port) or `"direct"` (only when the security group allows `host_port` from the deploying machine — ask, never assume). |
| Local tunnel port | `targets.ec2.local_tunnel_port` (ships `3230`) | Only for tunnel verify; change it if 3230 is taken locally. |

Arch matching needs no extra question: preflight compares each host's
`uname -m` against `image.platform` (or the local build arch) and fails
before anything mutates.

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

That must report **no differences at all** at this point. After rendering,
the only differences may be `deploy.config.json`, `README.md`, and the added
`k8s/` directory — every `.py` file stays byte-identical forever.

Then render placeholders in **exactly two files**: `deploy/deploy.config.json`
and `deploy/README.md`. Substitute only there, file by file — **never run a
blanket sed across the bundle**: the `.py` files must stay byte-identical to
the templates, and some contain literal `{{SERVICE_NAME}}`-style text in
comments that a global substitution would corrupt. JSON gotchas:

- The template config ships **both** target blocks under `targets`, in the
  order `"ec2"` then `"k8s"`. **Delete the block for a target the user did
  not select** — a leftover `{{...}}` placeholder anywhere makes the config
  fail to load (exit 2), by design.
- **When you delete a JSON block, its comma goes with it — and the LAST
  remaining member must have no trailing comma.** JSON has no trailing
  commas, so a careless deletion produces a file `json.load` rejects (exit
  2 before any check runs). Mechanically, for the two `targets` blocks:
  - **k8s-only bundle** (delete `"ec2"`, the first block): delete from the
    `"ec2": {` line through its closing `},` line **inclusive of that
    comma**. `"k8s": {` is then the first and only member — nothing else to
    fix. This is why `ec2` ships first: the common prune is comma-safe.
  - **ec2-only bundle** (delete `"k8s"`, the last block): delete from the
    `"k8s": {` line through its closing `}` line, **then remove the comma
    that now trails the `},` closing the `"ec2"` block** so it reads `}`.
    Skipping that second half is the classic failure — it yields
    `"ec2": {...},\n  }` which is invalid JSON.
  - The same rule applies to **array** elements, e.g. dropping the
    dependency object from `targets.k8s.services` (see below): delete the
    element together with the comma separating it from its neighbour, and
    leave no comma after the final element.
  - **Check it parses as soon as it can.** `{{INFERENCE_BODY}}` is the only
    placeholder that is NOT inside a JSON string, so the template cannot
    parse until you substitute it — an unsubstituted `{{INFERENCE_BODY}}` is
    an expected parse error, not a comma bug. So: do the block deletion,
    substitute `{{INFERENCE_BODY}}` (next bullet), then **immediately** run
    `python3 -m json.tool deploy/deploy.config.json >/dev/null` and fix it
    until it parses — before rendering the remaining placeholders. Do not
    defer this to the Step 2 validation at the end of this list: a comma
    mistake found now is one edit, found later it is buried under a dozen.
- Prune the README the same way: drop the chapters, CI/CD jobs, and rollback
  sections for targets that were not generated, and **retarget the generic
  examples** to the target(s) you did generate: the Usage line, the CI
  one-liners, the sample JSON summary, and the GitLab CI chapter's concrete
  deploy job in the template are all k8s-flavored (`--target k8s`,
  `"target": "k8s"`). For an ec2-only bundle, rewrite those to ec2 (adapt
  the GitLab job per the chapter's own notes: SSH key from a CI secret
  file-variable, no kubectl/kubeconfig, no dind needed when `--skip-build`).
  **Where to stop:** prune only target-*specific* prose — chapters, CI jobs,
  rollback recipes, and the secrets-wiring rows for a target you did not
  generate. **Keep the shared reference material even when it names the
  other target**: the `--target {k8s,ec2}` flag row, the
  `BENTOML_DEPLOY_EC2_HOSTS` environment override, and the stage-naming
  sentence listing `k8s.rollout[<slug>]` and `ec2.deploy[<host>]`. Those
  document the script's full, unchanging contract — the same `deploy.py`
  ships in every bundle — and editing them would misdescribe it. A few
  residual mentions of the other target in that reference material are
  correct, not leftovers.
- `{{INFERENCE_BODY}}` is substituted with a JSON object (no quotes), e.g.
  `{"text": "A great day"}`.
- **`targets.k8s.services` is a list, not a fixed pair.** The template ships
  a two-element example (one dependency + the entry service) purely to show
  the shape. Render **one object per BentoML service, in rollout order:
  dependencies first, entry last** — add elements for a three-service bento,
  and for a **single-service** bento delete the `{{DEP_SERVICE_*}}` element
  (with its trailing comma — see the JSON deletion rule above) so exactly
  one element remains with `"entry": true`. Exactly one element
  may have `"entry": true`; `name`/`slug` must both be unique and each slug
  must match the manifest filenames rendered in Step 3. The config's
  `"//services"` key is a free-text note (JSON has no comments; keys
  starting with `//` are ignored by the loader) — keep or delete it.
- The README's **Rollback (k8s)** block uses the same
  `{{DEP_SERVICE_SLUG}}` / `{{ENTRY_SERVICE_SLUG}}` placeholders: render one
  `rollout undo` + `rollout status` pair per service in the same order, and
  delete the "dependencies first" pair (and its comment) for a
  single-service bento.
- `schema` must stay `"bentoml-deploy-config/v2"`. A `v1` config (the old
  `deployment_name`/`service_name` shape) is rejected with exit 2 and a
  message telling the user to regenerate the bundle; there is no in-place
  migration, because the manifest layout changed with it.
- ec2 specifics: `hosts` is a JSON array of strings (render one
  `{{EC2_HOST}}` entry per instance); `env_names` stays `[]` unless the
  service needs runtime env vars (names only — remind the user the values
  come from the deploying environment); `registry_auth` and `verify_via`
  per the Step 1 table. `ssh_key_path` must point at the key where it
  already lives — never move or copy a private key into the project.
- Optional keys ship as JSON `null` (`project.service`, `image.platform`,
  `targets.k8s.image_pull_secret`, `targets.k8s.node_port`) or `false`
  (`image.local_image_preloaded`) — replace the null/false with the real
  value only when the parameter applies (no quotes around `null`, numbers,
  or booleans). With `registry_type: "none"` set `registry` to `""` and
  `local_image_preloaded` to `true` once the user has loaded the image.
- Validate: `python3 -m json.tool deploy/deploy.config.json >/dev/null`
  (parses — catches every comma mistake above),
  `find deploy -name __pycache__ -o -name '*.pyc'` (empty — no caches came
  along), and `grep -nE '\{\{[A-Z][A-Z0-9_]*\}\}' deploy/deploy.config.json deploy/README.md`
  must find nothing (README placeholders must be substituted too; the
  pattern matches only generator placeholders, not the `${{ secrets.* }}`
  expressions that belong in the README's GitHub Actions examples).

## Step 3 — Render the k8s manifests into `deploy/k8s/` (k8s target only)

The ec2 target needs no manifests — skip this step entirely for an
ec2-only bundle (and do not create `deploy/k8s/`).

The manifests come from the **`bentoml-k8s-deploy` skill's templates and
rules** — render them exactly as that skill's manifest-rendering step says
(per-service `args`/`--service-name`, the entry service's
`BENTOML_SERVE_DEPENDS`, labels/selectors, resources, probes, exposure,
pruning OPTIONAL blocks, no `{{...}}` left). **Do not restate or reinvent
those rules here**, and never hand-write manifests from memory. This skill
adds exactly one difference:

- **the image field is rendered as the literal sentinel
  `__DEPLOY_IMAGE__`** (not a real image ref) in *every* Deployment.
  `deploy.py` rewrites the sentinel in memory on every run and pipes the
  manifests to `kubectl apply -f -`, so the files on disk stay tag-free and
  committable.

What this bundle's preflight requires of the result (the layout the
`bentoml-k8s-deploy` templates already produce):

- Output dir `deploy/k8s/` (matching `targets.k8s.manifests_dir`), holding
  **one `<slug>-deployment.yaml` and one `<slug>-service.yaml` per service**
  in `targets.k8s.services`, with the same slugs. Extra files are fine and
  are applied too; `<slug>-hpa.yaml` is optional and never required.
- The `__DEPLOY_IMAGE__` sentinel as the **whole value of every real
  `image:` line** in every `<slug>-deployment.yaml`. Preflight matches the
  line, not the substring, so a sentinel mentioned only in a header comment
  does not satisfy it and a hand-pinned `image: registry/repo:tag` is
  rejected (it would deploy something other than what the run reports).
- Every object declaring `metadata.namespace` equal to
  `targets.k8s.namespace` — including each document of a multi-document
  file. A missing `namespace:` is rejected too: such an object would land in
  whatever namespace the kubeconfig currently points at while
  `rollout status -n <ns>` watches elsewhere.
- **Each dependency wired over the network**: for every non-entry service,
  some Deployment's `BENTOML_SERVE_DEPENDS` must contain the exact pair
  `<ServiceName>=http://<slug>.<namespace>.svc.cluster.local:3000` (the
  depending service's own Deployment — in a three-tier bento the middle
  service carries its own). Pairs are whitespace-separated, exactly one `=`
  each. No `BENTOML_RUNNER_MAP` / `BENTOML_SERVE_RUNNER_MAP` anywhere.
  Getting this wrong is not a loud failure: BentoML instantiates the
  dependency in-process with no log, and the deploy looks completely
  healthy — which is why preflight refuses it.
- **Non-entry `<slug>-service.yaml` must be `type: ClusterIP` with no
  `nodePort:`.** Inter-service traffic is unauthenticated pickle, so an
  exposed dependency port is remote code execution.
- No `spec.replicas` in a Deployment that has a matching `<slug>-hpa.yaml`.
- A fixed NodePort, if chosen, as a `nodePort: <value>` line in the **entry**
  service's `<slug>-service.yaml`, matching `targets.k8s.node_port`.
- For images only loaded into kind/minikube (registry_type `none`, or an
  image loaded by hand), `imagePullPolicy: IfNotPresent` under each
  container's `image:` line, per the k8s-deploy skill's private-registries
  reference.
- Sanity check: `grep -rn '{{' deploy/k8s/` empty, `ls deploy/k8s/` shows
  the expected per-service pair(s), and `grep -rln '__DEPLOY_IMAGE__'
  deploy/k8s/` lists every `*-deployment.yaml` and nothing else.

## Step 4 — Prove the bundle works, then hand it over

**Commit the bundle first** (`git add deploy/ k8s/ && git commit`). The image
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

1. **What was generated**: `deploy/deploy.py`, `deploy/deploy.config.json`
   (the only file they edit), `deploy/_internal/` (never edit),
   `deploy/k8s/*.yaml` — one Deployment + Service per BentoML service, the
   real deployment config for workload shape (k8s target only) —
   `deploy/README.md`.
2. **How the k8s run behaves** (k8s target only): the whole `deploy/k8s/`
   directory is applied in one pass, then each service's Deployment is
   waited on in `targets.k8s.services` order — dependencies first, entry
   last, because the entry service's `/readyz` fans out to its dependencies
   — and verification goes through the **entry** service's Service only.
   Each service gets its own `k8s.rollout[<slug>]` stage in the JSON summary.
   Verify then proves each dependency actually served a request (its own
   request counter must move); tell the user that this is what catches a
   dependency BentoML silently ran in-process, and that the run would
   otherwise look perfectly healthy. If
   they later add or remove a service, both the manifests and
   `targets.k8s.services` must change together (regenerate with this skill).
3. **How to deploy**: `python3 deploy/deploy.py --target <k8s|ec2>`
   (full build+push+deploy+verify), and the `--skip-build --image REF` form
   for redeploys/rollbacks. For ec2, secrets named in `env_names` must be
   exported in the deploying shell/CI environment first.
4. **Commit it**: the bundle contains no secrets —
   `git add deploy/ && git commit -m "Add production deploy bundle"`.
5. **Wire CI later**: point them at the CI/CD chapter in the generated
   `deploy/README.md` (complete GitHub Actions workflow with AWS OIDC and
   per-target deploy jobs, a GitLab CI equivalent, and
   `--check-only --local-only` as the fork-safe PR gate).
