# BentoML → Kubernetes Agent Skills

A set of [Claude Code agent skills](https://code.claude.com/docs/en/skills) for deploying
BentoML services to **your own Kubernetes cluster** using a fully open-source stack:
the `bentoml` CLI, Docker, and `kubectl` with plain Kubernetes manifests. No Helm charts,
no operators or CRDs, no closed-source dependencies, and no BentoCloud.

These skills cover **basic deployment**. Features that were part of the commercial
BentoCloud platform — scale-to-zero, inference-metric autoscaling, canary rollouts,
model registry sync, observability dashboards — are out of scope. For horizontal
scaling, standard Kubernetes HPA works with the manifests these skills produce.

## The skills

| Skill | What it does |
|---|---|
| [`bentoml-containerize`](bentoml-containerize/SKILL.md) | Turns a local BentoML project into a Docker image: verifies the runtime spec, `bentoml build`, `bentoml containerize`, local smoke test, push to your registry (Docker Hub, GHCR, ECR, private, `kind`/`minikube` load, or ttl.sh for throwaway tests). |
| [`bentoml-k8s-deploy`](bentoml-k8s-deploy/SKILL.md) | Deploys a pushed image to your cluster: gathers parameters, renders plain manifests (Deployment + Service, optional Ingress) into your project's `k8s/`, applies them, and verifies with a real inference request. |
| [`bentoml-k8s-troubleshoot`](bentoml-k8s-troubleshoot/SKILL.md) | Diagnostic runbook for failed deployments: ImagePullBackOff, CrashLoopBackOff, OOM, Pending/GPU, probe failures, unreachable services, inference errors. |

A typical session chains them: containerize → deploy → (troubleshoot if needed).

## Installation

Copy the skill directories to where Claude Code discovers skills:

```bash
# For all your projects (personal):
cp -r bentoml-containerize bentoml-k8s-deploy bentoml-k8s-troubleshoot ~/.claude/skills/

# Or for a single project (shared with your team via git):
cp -r bentoml-containerize bentoml-k8s-deploy bentoml-k8s-troubleshoot <your-project>/.claude/skills/
```

Then ask Claude Code things like "containerize my BentoML service and deploy it to my
Kubernetes cluster" — the skills activate on matching requests, or invoke them directly
with `/bentoml-containerize` etc.

## Prerequisites

- Python with `bentoml` ≥ 1.4 (`pip install bentoml`)
- Docker (daemon running) — used to build and smoke-test images
- `kubectl` configured with a context for your target cluster
- A container registry your cluster can pull from (or a local kind/minikube cluster)

## Conventions the skills follow

- BentoML serves on port **3000**; probes use **`/livez`** (liveness) and **`/readyz`**
  (readiness), with a generous startupProbe for model loading.
- Manifests are written to your project's `k8s/` directory and labeled
  `app.kubernetes.io/name: <service>` + `app.kubernetes.io/managed-by: bentoml-k8s-deploy`.
- Secrets are created imperatively (`kubectl create secret`); no secret material is ever
  written into manifest files.
- The deploy skill pins `--context` on every `kubectl` command; the troubleshoot skill
  confirms the current context before running anything. The skills always confirm the
  target cluster and namespace with you before applying anything.
