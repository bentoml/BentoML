# Pulling BentoML images from private registries (and local clusters)

The kubelet pulls the image, not the user's docker login — a working `docker pull` on the
laptop proves nothing about the cluster. Private registries need an `imagePullSecrets`
entry in the pod spec pointing at a `kubernetes.io/dockerconfigjson` Secret.

Never write registry credentials into YAML files. Always create the Secret imperatively:

```bash
kubectl --context <ctx> create secret docker-registry <bento-slug>-regcred -n <ns> \
  --docker-server=<REGISTRY> \
  --docker-username=<USERNAME> \
  --docker-password="$REGISTRY_TOKEN"
```

Then set `kubernetes.image_pull_secret: <bento-slug>-regcred` in `config.yml`; the
renderer puts `imagePullSecrets` on **every** Deployment, since all services of a bento run
the same image and need the same pull secret. The Secret is namespaced — recreate it in
every namespace that pulls the image.

Note the two halves are independent. `kubernetes.image_pull_secret` is how the **cluster**
pulls. How the **build machine** pushes is derived from the `image` URL and has no config
key: an `*.dkr.ecr.<region>.amazonaws.com` host triggers `aws ecr get-login-password` and a
describe-or-create of the repository; any other host assumes you have already run
`docker login`. There is no `image.registry_type` / `image.ecr_region` in the v4 schema, and
`image` never carries a tag (the tag is the bento version).

Quick test that credentials + image ref are right (run once, then delete):

```bash
kubectl --context <ctx> run pull-test -n <ns> --image=<IMAGE> \
  --overrides='{"spec":{"imagePullSecrets":[{"name":"<bento-slug>-regcred"}]}}' \
  --restart=Never --command -- sleep 300
kubectl --context <ctx> wait --for=condition=Ready pod/pull-test -n <ns> --timeout=60s \
  && echo "pull OK"   # times out -> check: kubectl --context <ctx> describe pod pull-test -n <ns>
kubectl --context <ctx> delete pod pull-test -n <ns>
```

## Registry specifics

### Docker Hub
- `--docker-server=https://index.docker.io/v1/`
- Use an access token (hub.docker.com → Account Settings → Personal access tokens), not
  the account password.
- Public Docker Hub images need no pull secret, but anonymous pulls are rate-limited
  (can cause intermittent `ImagePullBackOff` on busy clusters — a pull secret with a free
  account raises the limit).

### GitHub Container Registry (GHCR)
- `--docker-server=ghcr.io`, `--docker-username=<github-username>`,
  `--docker-password=$GITHUB_PAT` (classic PAT with `read:packages`).
- Packages are private by default even in public repos — check the package's visibility
  before assuming anonymous pulls work.

### AWS ECR
- Tokens are **temporary (12 h)** — fine for a one-off deploy, but the Secret must be
  refreshed for future pulls (node restarts, scale-ups):

```bash
kubectl --context <ctx> create secret docker-registry <bento-slug>-regcred -n <ns> \
  --docker-server=<ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com \
  --docker-username=AWS \
  --docker-password="$(aws ecr get-login-password --region <REGION>)"
```

- On EKS, prefer the node IAM role with the `AmazonEC2ContainerRegistryReadOnly` policy —
  then **no pull secret is needed at all**; delete the `imagePullSecrets` block.

### Self-hosted / other (Harbor, GitLab, Quay, ...)
- Same `docker-registry` Secret pattern with the registry's hostname.
- HTTP-only or self-signed registries fail with x509/`http: server gave HTTP response`
  errors; fixing that requires containerd config on every node — out of scope; recommend
  a registry with valid TLS.

## Local clusters: skip the registry entirely

If the image exists in the local docker daemon and the cluster is kind/minikube/k3d, load
it directly instead of pushing:

```bash
kind load docker-image <IMAGE> --name <kind-cluster-name>   # kind
minikube image load <IMAGE>                                  # minikube
k3d image import <IMAGE> -c <cluster-name>                   # k3d
```

Then in `config.yml` set `image: ""` (an empty image URL is what says "the image is
already on the nodes") and leave `kubernetes.image_pull_secret: null`. Nothing is pushed,
the bento tag is used as the image name, and the rendered `imagePullPolicy: IfNotPresent`
keeps the kubelet from trying to pull an image that only exists on the node — avoid mutable
`:latest`-style tags here, since a stale local layer can then be used silently.

## Throwaway public registry: ttl.sh

For quick tests with zero setup, ttl.sh is an anonymous, ephemeral public registry
(the tag is the time-to-live):

```bash
docker tag <IMAGE> ttl.sh/<any-unique-name>:1h
docker push ttl.sh/<any-unique-name>:1h
```

Render with `--image ttl.sh/<any-unique-name>:1h` (in `config.yml`:
`image: ttl.sh/<any-unique-name>`, no tag — the tag ttl.sh reads as its TTL is the bento
version, so pass `--version 1h` if you want the bundle to build and push it itself); no
pull secret needed. The registry host is not an ECR host, so no login is attempted, which
is right for ttl.sh. The image vanishes after the TTL and is publicly pullable meanwhile —
never use it for anything sensitive or lasting.
