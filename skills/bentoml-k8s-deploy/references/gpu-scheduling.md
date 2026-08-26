# GPU scheduling for BentoML services

GPUs are requested through the extended resource `nvidia.com/gpu`, which you add to
`services.<Name>.resources.limits` in `config.yml` (per service — request GPUs only for the
services that need them):

```yaml
    resources:
      requests: {cpu: "2", memory: "8Gi"}
      limits:   {cpu: "4", memory: "16Gi", nvidia.com/gpu: "1"}
```

Extended resources cannot be fractional and requests default to limits — setting the limit
alone is correct. Quoted, like every quantity.

## Precondition: the cluster must advertise GPUs

The **NVIDIA device plugin** (usually via the NVIDIA GPU Operator or the plugin
DaemonSet) must already be running — installing it is the cluster admin's job and out of
scope for this skill. Verify before writing a GPU manifest:

```bash
kubectl --context <ctx> get nodes \
  -o custom-columns='NAME:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu'
```

Every value is `<none>` or `0` → no schedulable GPUs (`0` means the device plugin runs
but advertises no GPUs on that node); a GPU pod will sit **Pending** forever with
`Insufficient nvidia.com/gpu`. Tell the user and either deploy CPU-only or stop.

Also check free capacity — allocatable minus what other pods already claim:

```bash
kubectl --context <ctx> describe node <gpu-node> | grep -A8 'Allocated resources'
```

## What the pod gets

- `nvidia.com/gpu: 1` grants exclusive use of one physical GPU (unless the admin
  configured time-slicing/MIG — then one "gpu" is a slice; ask the admin, don't assume).
- The device plugin injects the devices and driver libraries; the container needs no
  privileged mode and no host mounts.
- Multi-GPU (`nvidia.com/gpu: 2+`) only helps if the serving framework uses multiple
  GPUs (e.g. vLLM tensor parallelism). Default to 1 per pod.

## The image must be CUDA-capable

Requesting a GPU does not make the code use it. The image needs CUDA-enabled ML
libraries — in BentoML that comes from the project's image spec
(`bentofile.yaml` `docker.cuda_version`, or a `bentoml.images.Image` runtime built on a
CUDA base). If the containerize step produced a CPU-only image, the pod will schedule
fine but run on CPU (or crash importing CUDA libs). When in doubt, check the logs after
startup for the framework's device report (e.g. `torch.cuda.is_available()`).

## Taints, tolerations, node selection

GPU nodes are commonly tainted so CPU pods don't land on them. If GPU pods stay Pending
with `node(s) had untolerated taint`, check the taint and add matching entries to that
service's `tolerations:` in `config.yml` (verbatim Kubernetes syntax, rendered into the pod
spec):

```bash
kubectl --context <ctx> describe node <gpu-node> | grep -i taint
```

```yaml
      tolerations:
        - key: nvidia.com/gpu        # match the actual taint key from the node
          operator: Exists
          effect: NoSchedule
```

To pin to a specific GPU type on heterogeneous clusters, add a nodeSelector using a label
that exists on the nodes (check `kubectl --context <ctx> get nodes --show-labels`; common ones:
`nvidia.com/gpu.product`, cloud instance-type labels):

```yaml
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-A10G
```

## Sizing reminders

- Keep CPU/memory realistic alongside the GPU: model loading and tokenization still use
  host RAM; an OOMKilled GPU pod wastes the whole GPU. For LLM-sized models raise memory
  (e.g. request 8Gi / limit 16Gi) rather than reusing the CPU defaults.
- Startup is slower on GPU nodes (image is bigger, weights move to VRAM). The default
  10-minute startupProbe budget usually suffices; for very large models raise
  `probes.startup_failure_threshold` (periodSeconds is 10, so 60 → 10 min, 120 → 20 min,
  240 → 40 min) and raise `kubernetes.rollout_timeout_seconds` past it — it must stay
  LONGER than the startup budget, or a pod that uses its whole budget loses the race and
  the rollout is reported failed just as it succeeds.
- Replicas × GPUs-per-pod must fit the cluster's free GPU count, or the extra replicas
  stay Pending.
