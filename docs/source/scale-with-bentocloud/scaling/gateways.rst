===========================================
Scale Across Multiple Regions with Gateways
===========================================

Modern GPU operations are hindered by fragmented suppliers with inconsistent
environments, variable cost and reliability, unpredictable access to committed
versus on-demand capacity, and limited regional GPU availability, all of which
make it difficult for teams to provision efficiently, avoid vendor lock-in, and
reliably scale inference workloads without overpaying or encountering capacity
shortages.

Gateways provide a unified abstraction for operating **distributed GPU clusters**
across clouds, regions, and providers. With BentoCloud Gateways, teams can scale
inference workloads elastically across heterogeneous GPU infrastructure while
exposing a **single, stable endpoint** to clients. Gateways make it possible to
treat GPUs from hyperscalers and neoclouds as
**one logical, multi-region GPU cluster**, enabling high availability and
cost-efficient scaling without operational complexity.



How Gateways Work
=================

Consistent Endpoint URL
-----------------------

Each Gateway exposes a single HTTPS endpoint. BentoCloud routes requests to the
optimal upstream deployments based on model name, request parameters, or user-defined
routing policies.

Heterogeneous Cluster Abstraction
---------------------------------

Gateways unify diverse infrastructure types, managed Kubernetes clusters,
bare-metal machines, virtual machines, and neocloud GPU providers, under a
consistent, normalized execution environment. BentoCloud abstracts GPU SKUs into
capacity units based on real throughput, enabling predictable scheduling across
heterogeneous fleets.

Vendor-Agnostic Routing
-----------------------

Gateways decouple inference endpoints from any specific provider, allowing you
to:

* Mix hyperscalers and neoclouds behind one endpoint.
* Avoid infrastructure or vendor lock-in.
* Add or remove regions without requiring changes from clients.
* Maintain seamless failover when regional outages occur.

Multi-Region Elasticity
-----------------------

Gateways automatically use committed GPUs for baseline workloads and burst into
other regions' elastic compute pools when demand exceeds local
capacity. This ensures high availability and smooth handling of traffic spikes.


Creating a Gateway
==================

To create a Gateway, configure the following fields either in the BentoCloud
Console or programmatically via the API.

Name
----

The Gateway name becomes the prefix of the public URL ``<name>.example.com``.

Domain
------

The domain forms the suffix of the public endpoint ``name.<example.com>``.

Protocol
--------

The protocol defines how requests are interpreted and routed. For example, with
the ``OpenAI Chat Completions`` protocol, routing is based on the ``model`` field
in the request—only deployments that support the requested model are eligible to
serve the traffic.

Load Balancing Strategy
-----------------------

**Overflow Routing**

   Routes requests to the highest-ranked deployment first. Once its capacity is
   reached, overflow traffic proceeds to the next deployment.

**Capacity-based Round-Robin**

   Distributes requests proportionally according to each deployment's currently
   available capacity.

Upstream Deployments
--------------------

Upstream deployments represent the set of deployments routed behind a
Gateway. These can span multiple regions and cloud providers. Gateways route
traffic to these deployments according to the configured protocol and load
balancing strategy.


Summary
=======

Gateways transform fragmented, region-bound GPU infrastructure into a unified,
global inference fabric. By abstracting vendor differences, normalizing GPU
capacity, and enabling multi-region elasticity, they allow teams to scale
reliably without operational overhead.
