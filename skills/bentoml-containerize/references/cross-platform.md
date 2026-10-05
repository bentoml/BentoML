# Cross-platform (cross-architecture) image builds

## When this matters

The image must match the **cluster nodes'** CPU architecture, not the build machine's.
Typical mismatch: Apple Silicon (arm64) building for a cloud cluster (amd64). Symptom in
`kubectl logs`, the pod crashes instantly with
`exec /home/bentoml/bento/env/docker/entrypoint.sh: exec format error` or similar.

## Detecting architectures

```bash
docker info --format '{{.Architecture}}'      # build machine: x86_64 => amd64, aarch64 => arm64
# Cluster nodes (needs kubectl; else ask the user):
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.nodeInfo.architecture}{"\n"}{end}'
```

## Single target platform

Platform is a backend option:

```bash
bentoml containerize "$BENTO_TAG" -t "$IMAGE" --opt platform=linux/amd64
```

`--platform=linux/amd64` is the older spelling: it maps to `--opt platform=...` and
prints a deprecation-style warning; both work.

- Emulated builds (arm64 host, amd64 target) go through QEMU/Rosetta and are **much
  slower** — minutes even for small services. Docker Desktop handles it out of the box;
  plain Linux may need binfmt handlers:
  `docker run --privileged --rm tonistiigi/binfmt --install all`.
- You generally **cannot smoke-test** a cross-arch image locally: Docker Desktop runs
  amd64 under emulation (slowly), plain Linux cannot without those handlers. Skip the
  local `docker run` smoke test; verification happens after deployment.
- Not `bentoml build --platform`, which sets the platform for **Python dependency
  locking** (values like `linux`, `x86_64-unknown-linux-gnu`), not the image arch. When
  cross-building from macOS it also helps locked wheels resolve for Linux.

## Multi-arch images (amd64 + arm64 in one tag)

Only for one tag serving mixed-arch clusters. Needs the buildx backend and pushes
straight to the registry (multi-arch manifests cannot be loaded into the local image
store), so log in first — `--opt push` uploads as part of the build:

```bash
docker buildx create --use --name bento-builder 2>/dev/null || docker buildx use bento-builder
bentoml containerize "$BENTO_TAG" -t "$IMAGE" \
  --backend buildx \
  --opt platform=linux/amd64 --opt platform=linux/arm64 \
  --opt push

docker manifest inspect "$IMAGE" | grep -E 'architecture|os'   # verify
```

For a basic single-cluster deployment prefer single-platform: simpler, faster.
