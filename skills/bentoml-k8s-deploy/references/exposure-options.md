# Exposure options for a BentoML service on Kubernetes

The container always listens on **3000**, and the rendered Service exposes 3000 (named
`http`) in every mode. Exposure lives in `config.yml` under the **entry** service's
`expose:`/`ingress:` block — change it, re-render, re-apply. Only the entry service is ever
exposed: dependency Services stay `ClusterIP` because inter-service traffic is
`application/vnd.bentoml+pickle` (unauthenticated pickle deserialization), so
`expose:`/`ingress:` elsewhere is a config error. The entry service is `bento.yaml`'s
`entry_service`, not a config key; `<entry-slug>` below is its slug.

| Method | Reachable from | Needs | Best for |
|---|---|---|---|
| port-forward | your machine only | nothing | development, testing, demos |
| ClusterIP (default) | inside the cluster | nothing | other in-cluster workloads calling the model |
| NodePort | anyone who can reach a node IP | routable node IPs | bare-metal / homelab without a LB |
| LoadBalancer | internet / VPC | cloud LB integration (EKS/GKE/AKS) or MetalLB | production on managed clouds |
| Ingress | internet, name-based | an installed ingress controller | production with hostnames/TLS, several services behind one IP |

## port-forward (default for testing)

`expose: {type: ClusterIP}` (the default); nothing is reachable from outside.

```bash
kubectl --context <ctx> port-forward svc/<entry-slug> -n <ns> 3100:3000
# then: curl http://127.0.0.1:3100/readyz
```

The tunnel lives only while the command runs. Forward to an **uncommon** local port like
3100 rather than 3000: if a dev server already holds 3000 the forward fails to bind and
your curls silently hit that process instead, producing convincing but fake results
(same reasoning as the verification step in SKILL.md). Always confirm the port-forward
process is still alive before trusting a response.

## ClusterIP (in-cluster consumers)

Other pods call the service at
`http://<slug>.<ns>.svc.cluster.local:3000` — this is also exactly the form used in
`BENTOML_SERVE_DEPENDS` to wire one BentoML service to another.

## NodePort

`expose: {type: NodePort}` on the entry service. Kubernetes assigns a port in
30000–32767; pin it with `node_port:` in the same block to keep the URL stable:

```bash
kubectl --context <ctx> get svc <entry-slug> -n <ns> \
  -o jsonpath='{.spec.ports[0].nodePort}'
kubectl --context <ctx> get nodes -o wide     # take any node's EXTERNAL-IP (or INTERNAL-IP on a LAN)
curl http://<node-ip>:<node-port>/readyz
```

Caveats: node IPs may not be routable from the user's network (cloud security groups /
firewalls must allow the port range); the port changes if the Service is recreated. For
kind clusters NodePort is not reachable from the host unless the port was mapped in the
kind config — prefer port-forward there.

## LoadBalancer

`expose: {type: LoadBalancer}` on the entry service (plus `annotations:` there for
provider-specific tuning). Only works where something provisions LBs
(managed clouds; MetalLB on bare metal). Wait for the address:

```bash
kubectl --context <ctx> get svc <entry-slug> -n <ns> -w
# EXTERNAL-IP goes from <pending> to an IP/hostname; then:
curl http://<external-ip>:3000/readyz
```

If EXTERNAL-IP stays `<pending>` for minutes, the cluster has no LB provider — fall back
to NodePort or Ingress. Warn the user this usually creates a **billable, publicly
reachable** cloud load balancer with no auth in front of the model.

## Ingress

`ingress: {enabled: true, class_name: ..., host: ...}` on the entry service, normally with
`expose: {type: ClusterIP}`. Precondition — a controller must already be installed:

```bash
kubectl --context <ctx> get ingressclass
```

No output → no controller; installing one (ingress-nginx, traefik, cloud-specific) is out
of scope for this skill — offer NodePort/LoadBalancer instead. Otherwise use the listed
name as `ingress.class_name` (a class marked `(default)` still should be set explicitly).

- `ingress.host`: a DNS name the user controls, pointed at the controller's external
  address (`kubectl --context <ctx> get svc -n ingress-nginx`, or the controller's
  namespace). For quick
  tests without DNS: `curl -H 'Host: <host>' http://<controller-ip>/readyz` or an
  `/etc/hosts` entry.
- **TLS**: set `ingress.tls_secret` only if that TLS Secret exists in the same
  namespace — via cert-manager (out of scope to install) or manually:
  `kubectl --context <ctx> create secret tls <name> -n <ns> --cert=cert.pem --key=key.pem`.
  Leave it null to serve plain HTTP; the renderer then emits no `tls:` block.
- **Timeouts**: inference can exceed a controller's default upstream timeout (often 60s
  for nginx). For long-running requests on ingress-nginx add annotations such as
  `nginx.ingress.kubernetes.io/proxy-read-timeout: "300"` and
  `nginx.ingress.kubernetes.io/proxy-body-size: "50m"` (large payloads). Other
  controllers have equivalents — check theirs before debugging 504s.

## A note on exposure and safety

NodePort/LoadBalancer/Ingress make the model API reachable by anyone who can reach the
address — BentoML applies no authentication by itself. Remind the user to put auth
(ingress auth annotations, an API gateway, or network policy/firewall rules) in front of
anything non-experimental.
