# Cross-platform (cross-architecture) image builds

## When this matters

The image must match the CPU architecture of the **cluster nodes**, not the
build machine. Typical mismatch: building on Apple Silicon (arm64) for a cloud
cluster (amd64). Symptom of getting it wrong: the pod crashes instantly with
`exec /home/bentoml/bento/env/docker/entrypoint.sh: exec format error` (or
similar `exec format error`) in `kubectl logs`.

## Detecting architectures

```bash
# Build machine / docker daemon
docker info --format '{{.Architecture}}'      # x86_64 => amd64, aarch64 => arm64

# Cluster nodes (needs kubectl access; otherwise ask the user)
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.nodeInfo.architecture}{"\n"}{end}'
```

## Single target platform

`bentoml containerize` accepts the platform via the backend option:

```bash
bentoml containerize "$BENTO_TAG" -t "$IMAGE" --opt platform=linux/amd64
```

`--platform=linux/amd64` is the older spelling of the same thing (it maps to
`--opt platform=...` and prints a deprecation-style warning; both work).

Notes:

- Emulated builds (arm64 host building amd64) go through QEMU/Rosetta and are
  **much slower** — expect several minutes even for small services. On Docker
  Desktop this works out of the box; on plain Linux you may need binfmt
  handlers: `docker run --privileged --rm tonistiigi/binfmt --install all`.
- You generally **cannot smoke-test** a cross-arch image on the build machine
  (Docker Desktop can run amd64 images under emulation, slowly; plain Linux
  cannot unless the binfmt/QEMU handlers shown above are installed). Skip the
  local `docker run` smoke test and note that verification happens after
  deployment instead.
- Do not confuse this with `bentoml build --platform` — that flag controls the
  platform used for **Python dependency locking** (values like `linux`,
  `x86_64-unknown-linux-gnu`), not the container image arch. When
  cross-building from macOS it can help to also pass `bentoml build
  --platform linux` so locked wheels resolve for Linux.

## Multi-arch images (amd64 + arm64 in one tag)

Only needed when the same tag must run on mixed-arch clusters. Requires the
buildx backend and pushes directly to the registry (multi-arch manifests cannot
be loaded into the local docker image store):

```bash
docker buildx create --use --name bento-builder 2>/dev/null || docker buildx use bento-builder

bentoml containerize "$BENTO_TAG" -t "$IMAGE" \
  --backend buildx \
  --opt platform=linux/amd64 --opt platform=linux/arm64 \
  --opt push
```

You must be logged in to the registry before running this (`--opt push` uploads
as part of the build). Verify with:

```bash
docker manifest inspect "$IMAGE" | grep -E 'architecture|os'
```

For a basic single-cluster deployment, prefer a single-platform build — it is
simpler and faster.
