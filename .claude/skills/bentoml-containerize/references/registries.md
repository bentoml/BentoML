# Registry-specific push instructions

Conventions used below:

- `$BENTO_TAG` — bento tag from the build step, e.g. `summarization:6oxk5qvott3lsnry`.
- Split it when you need the parts:
  ```bash
  BENTO_NAME=${BENTO_TAG%%:*}      # summarization
  BENTO_VERSION=${BENTO_TAG##*:}   # 6oxk5qvott3lsnry
  ```
- Image repository names must be **lowercase**. Reuse `$BENTO_VERSION` as the
  image tag for traceability.
- The sections below assume the SKILL.md Step 3 flow:
  `bentoml containerize "$BENTO_TAG" -t "$IMAGE"`. With `-t`, the image is
  tagged ONLY `$IMAGE` — **no image named `$BENTO_TAG` exists**. Compose
  `$IMAGE` from the section's naming pattern BEFORE containerizing.
- Only if you containerized **without** `-t` (image named `$BENTO_TAG`, e.g.
  the kind/minikube path), retag before pushing:
  `docker tag "$BENTO_TAG" "$IMAGE"`.
- After pushing, record `$IMAGE` for the deploy-skill handoff (Step 6).

## Docker Hub

```bash
docker login                       # prompts for Docker Hub username + password/PAT
IMAGE="docker.io/<dockerhub-username>/${BENTO_NAME}:${BENTO_VERSION}"
# (containerize with -t "$IMAGE"; if you containerized WITHOUT -t, first:
#  docker tag "$BENTO_TAG" "$IMAGE")
docker push "$IMAGE"
```

- Free accounts: pushes to a repo that doesn't exist create it as **public** by
  default (private repos are limited). If the repo must be private, the cluster
  needs an imagePullSecret (deploy skill handles it; it will need the username
  and a PAT).

## GHCR (GitHub Container Registry)

```bash
# PAT needs the write:packages scope (classic PAT), or use gh's token:
echo "$GITHUB_TOKEN" | docker login ghcr.io -u <github-username> --password-stdin
# With gh CLI installed: gh auth token | docker login ghcr.io -u <github-username> --password-stdin

IMAGE="ghcr.io/<github-username-or-org>/${BENTO_NAME}:${BENTO_VERSION}"
# (containerize with -t "$IMAGE"; if you containerized WITHOUT -t, first:
#  docker tag "$BENTO_TAG" "$IMAGE")
docker push "$IMAGE"
```

- **Gotcha:** new GHCR packages are **private by default**. Either make the
  package public (GitHub → Packages → package settings → Change visibility) or
  tell the deploy skill an imagePullSecret is required (username + PAT with
  `read:packages`).

## AWS ECR

If a deploy skill prescribes a repository name, use that name instead of the
plain `$BENTO_NAME` below.

```bash
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=<region>
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ECR requires the repository to exist before pushing. Create it only if
# missing — keep stderr visible so permission/name errors surface here,
# not as a confusing "name unknown" at push time:
aws ecr describe-repositories --repository-names "$BENTO_NAME" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$BENTO_NAME" --region "$AWS_REGION"

IMAGE="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${BENTO_NAME}:${BENTO_VERSION}"
# (containerize with -t "$IMAGE"; if you containerized WITHOUT -t, first:
#  docker tag "$BENTO_TAG" "$IMAGE")
docker push "$IMAGE"
```

- EKS nodes in the same account usually pull from ECR without imagePullSecrets
  (node IAM role). Non-EKS clusters need a docker-registry secret with
  temporary ECR credentials (they expire every 12h — mention this caveat).

## Generic private registry

```bash
docker login registry.example.com      # ask user for credentials
IMAGE="registry.example.com/<project>/${BENTO_NAME}:${BENTO_VERSION}"
# (containerize with -t "$IMAGE"; if you containerized WITHOUT -t, first:
#  docker tag "$BENTO_TAG" "$IMAGE")
docker push "$IMAGE"
```

- HTTP-only or self-signed registries need `insecure-registries` in the Docker
  daemon config on every machine that pulls — including the cluster nodes.
  Prefer a TLS-enabled registry.
- The cluster will need an imagePullSecret unless the nodes are pre-configured
  for this registry — pass this fact to the deploy skill.

## kind / minikube — no registry at all (local clusters)

Load the image straight into the cluster nodes; nothing is pushed. This path
assumes you containerized **without** `-t` (per SKILL.md Step 3), so the image
is named exactly `$BENTO_TAG`.

```bash
# kind (find the cluster name with: kind get clusters)
kind load docker-image "$BENTO_TAG" --name <cluster-name>

# minikube
minikube image load "$BENTO_TAG"
```

- The image reference for the deployment is the local name (e.g. `$BENTO_TAG`
  itself) — no registry prefix.
- **Critical for the deploy skill:** `imagePullPolicy` must be `IfNotPresent`
  (or `Never`). With `Always` — Kubernetes' default for `:latest`-style
  unqualified tags — the kubelet tries to pull from Docker Hub and fails.
- Loading multi-GB images into kind can be slow; that's expected.

## ttl.sh — anonymous ephemeral registry (throwaway tests)

No account, no login; images are world-readable and auto-expire. Never use for
anything sensitive. The **tag is the time-to-live** (`1h`, up to `24h`).
Note: the "reuse the bento version as the image tag" traceability rule does NOT
apply here — the tag must be the TTL and the repository name is random.

```bash
# Random repository name; falls back if uuidgen is not installed
RAND=$( (uuidgen 2>/dev/null || head -c16 /dev/urandom | od -An -tx1) | tr -d ' \n' | tr 'A-Z' 'a-z' )
IMAGE="ttl.sh/${RAND}:2h"
# (containerize with -t "$IMAGE"; if you containerized WITHOUT -t, first:
#  docker tag "$BENTO_TAG" "$IMAGE")
docker push "$IMAGE"
echo "Ephemeral image (expires in 2h): $IMAGE"
```

- Public and unauthenticated: anyone with the URL can pull. Fine for demo
  services; NOT for images containing private models or proprietary code.
- Deploy within the TTL window; the cluster cannot pull after expiry.
- **Handoff caveat:** when passing a ttl.sh image to `bentoml-k8s-deploy`, tell
  the deploy skill NOT to derive the Kubernetes Service/Deployment name from
  the image repository — it's a random UUID that may start with a digit
  (invalid DNS-1035 Service name). Pass the snake_cased bento name with
  underscores converted to hyphens (e.g. `my_service` → `my-service`) as the
  suggested service name instead.

## Verifying a push

```bash
docker manifest inspect "$IMAGE" > /dev/null && echo "image is pullable"
```

(For kind/minikube loads instead: `docker exec <kind-node> crictl images | grep "$BENTO_NAME"`
or `minikube image ls | grep "$BENTO_NAME"`.)
