# Registry-specific push instructions

Conventions:

- `$BENTO_TAG` — the tag from the build step, e.g. `summarization:6oxk5qvott3lsnry`:
  ```bash
  BENTO_NAME=${BENTO_TAG%%:*}      # summarization
  BENTO_VERSION=${BENTO_TAG##*:}   # 6oxk5qvott3lsnry
  ```
- Repository names must be **lowercase**; the tag must be `$BENTO_VERSION` — the deploy
  config appends the bento version itself (SKILL.md Step 3).
- Sections assume the Step 3 flow `bentoml containerize "$BENTO_TAG" -t "$IMAGE"`, which
  tags the image ONLY `$IMAGE` — **no image named `$BENTO_TAG` exists**. So compose
  `$IMAGE` from the pattern below BEFORE containerizing.
- Containerized **without** `-t` (image named `$BENTO_TAG`, e.g. kind/minikube)? Retag
  first: `docker tag "$BENTO_TAG" "$IMAGE"`.
- Record `$IMAGE` for the deploy-skill handoff (Step 6).

## Docker Hub

```bash
docker login                       # prompts for username + password/PAT
IMAGE="docker.io/<dockerhub-username>/${BENTO_NAME}:${BENTO_VERSION}"
docker push "$IMAGE"
```

- Free accounts: pushing to a nonexistent repo creates it **public** (private repos are
  limited); a private repo needs an imagePullSecret, which the deploy skill sets up from
  username + PAT.

## GHCR (GitHub Container Registry)

```bash
# PAT needs the write:packages scope (classic PAT), or use gh's token:
echo "$GITHUB_TOKEN" | docker login ghcr.io -u <github-username> --password-stdin
# Or with gh CLI: gh auth token | docker login ghcr.io -u <github-username> --password-stdin
IMAGE="ghcr.io/<github-username-or-org>/${BENTO_NAME}:${BENTO_VERSION}"
docker push "$IMAGE"
```

- **Gotcha:** new GHCR packages are **private by default**. Either make it public
  (GitHub → Packages → settings → Change visibility) or tell the deploy skill an
  imagePullSecret is needed (username + PAT, `read:packages`).

## AWS ECR

A deploy-skill-prescribed repository name overrides `$BENTO_NAME`.

```bash
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=<region>
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# The repository must exist before pushing. Create only if missing; keep stderr visible
# so permission/name errors surface here, not as "name unknown" at push:
aws ecr describe-repositories --repository-names "$BENTO_NAME" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$BENTO_NAME" --region "$AWS_REGION"

IMAGE="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${BENTO_NAME}:${BENTO_VERSION}"
docker push "$IMAGE"
```

- EKS nodes in the same account usually pull without imagePullSecrets (node IAM role).
  Non-EKS clusters need a docker-registry secret holding temporary ECR credentials —
  they expire every 12h; mention that caveat.

## Generic private registry

```bash
docker login registry.example.com      # ask user for credentials
IMAGE="registry.example.com/<project>/${BENTO_NAME}:${BENTO_VERSION}"
docker push "$IMAGE"
```

- HTTP-only or self-signed registries need `insecure-registries` in the Docker daemon
  config of every machine that pulls, cluster nodes included. Prefer TLS.
- The cluster needs an imagePullSecret unless its nodes are already configured for it —
  tell the deploy skill.

## kind / minikube — no registry at all (local clusters)

Load the image into the cluster nodes; nothing is pushed. Assumes no `-t`, so the image
is named exactly `$BENTO_TAG`.

```bash
kind load docker-image "$BENTO_TAG" --name <cluster-name>   # name from: kind get clusters
minikube image load "$BENTO_TAG"                            # minikube instead of kind
```

- The deployment's image reference is that local name — no registry prefix.
- **Critical for the deploy skill:** `imagePullPolicy` must be `IfNotPresent` (or
  `Never`). With `Always` — Kubernetes' default for `:latest`-style unqualified tags —
  the kubelet pulls from Docker Hub and fails.
- Loading multi-GB images into kind is slow; expected.

## ttl.sh — anonymous ephemeral registry (throwaway tests)

No account, no login; images are world-readable and auto-expire. Never for anything
sensitive. The **tag is the time-to-live** (`1h`, up to `24h`), so the tag-is-the-bento-
version rule does not apply here; the repository name is random.

```bash
# Random repo name; fallback if uuidgen is missing
RAND=$( (uuidgen 2>/dev/null || head -c16 /dev/urandom | od -An -tx1) | tr -d ' \n' | tr 'A-Z' 'a-z' )
IMAGE="ttl.sh/${RAND}:2h"
docker push "$IMAGE"
echo "Ephemeral image (expires in 2h): $IMAGE"
```

- Anyone with the URL can pull: fine for demos, not for private models or proprietary
  code.
- Deploy within the TTL window; the cluster cannot pull after expiry.
- **Handoff caveat:** tell `bentoml-k8s-deploy` NOT to derive the Kubernetes
  Service/Deployment name from the image repository — a random UUID may start with a
  digit, invalid as a DNS-1035 Service name. Pass the snake_cased bento name,
  underscores as hyphens (`my_service` → `my-service`).

## Verifying a push

```bash
docker manifest inspect "$IMAGE" > /dev/null && echo "image is pullable"
```

For kind/minikube loads: `docker exec <kind-node> crictl images | grep "$BENTO_NAME"` or
`minikube image ls | grep "$BENTO_NAME"`.
