===========================================
Scale across multiple regions with Gateways
===========================================

Modern GPU operations are hindered by fragmented suppliers with inconsistent
environments, variable cost and reliability, unpredictable access to committed
versus on-demand capacity, and limited regional GPU availability. These issues
make it difficult to provision compute efficiently, avoid vendor lock-in, and
reliably scale inference without overpaying or encountering capacity shortages.

BentoCloud Gateways solve this by providing a unified abstraction for operating 
**distributed GPU clusters** across clouds, regions, and vendors. You can scale
inference elastically on mixed GPU fleets while exposing a **single, stable endpoint**
to your clients. This lets you treat GPUs from hyperscalers and neoclouds as
**one logical, multi-region GPU cluster**, enabling high availability and
cost-efficient scaling without operational complexity.

How Gateways work
-----------------

Gateways route requests to the best available Deployments while hiding infrastructure differences.

Consistent endpoint URL
^^^^^^^^^^^^^^^^^^^^^^^

Each Gateway exposes a single HTTPS endpoint. BentoCloud routes requests to the
optimal upstream Deployments based on model name, request parameters, or user-defined
routing policies.

Heterogeneous cluster abstraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gateways unify diverse infrastructure types under a consistent, normalized execution environment,
including:

* Managed Kubernetes clusters
* Bare-metal machines
* Virtual machines
* Neocloud GPU providers

BentoCloud abstracts GPU SKUs into capacity units based on real throughput,
ensuring predictable scheduling across heterogeneous fleets.

Vendor-agnostic routing
^^^^^^^^^^^^^^^^^^^^^^^

Gateways decouple inference endpoints from any specific provider, allowing you
to:

* Mix hyperscalers and neoclouds behind one endpoint.
* Avoid infrastructure or vendor lock-in.
* Add or remove regions without requiring changes from clients.
* Maintain seamless failover when regional outages occur.

Multi-region elasticity
^^^^^^^^^^^^^^^^^^^^^^^

Gateways automatically use committed GPUs for baseline workloads. When demand exceeds
local capacity, they burst into other regions that have available elastic capacity.
This ensures high availability and smooth handling of traffic spikes.

Creating a Gateway
------------------

To create a Gateway, configure the following fields either in the BentoCloud
Console or programmatically via the API.

* **Name**: The Gateway name becomes the prefix of the public endpoint ``<name>.example.com``.
* **Domain**: The domain forms the suffix of the public endpoint ``name.<example.com>``.
* **Protocol**: The protocol defines how BentoCloud interprets and routes requests. For example, with the `OpenAI Chat Completions` protocol, routing is based on the ``model`` field in the request; only Deployments that support that model receive the request.
* **Load balancing strategy**
    * **Overflow Routing**: Routes requests to the highest-ranked Deployment first. Once it reaches capacity, overflow traffic proceeds to the next Deployment.
    * **Capacity-Based Round Robin**: Distributes requests proportionally according to each Deployment’s currently available capacity.
* **Upstream Deployments**: The set of Deployments behind a Gateway. They can span multiple regions and cloud providers. Gateways route traffic to them based on the configured protocol and load balancing strategy.
