# Security Policy

## Supported Versions

BentoML is currently under active development and releases a new version
every 2-3 weeks. We always recommend users to move to a newer version
when it became available, and we only provide security updates in the
latest version.

If you are using an older version of BentoML and would like to receive
security patches, let us know via
[BentoML Slack Channel](https://join.slack.bentoml.org)
or [BentoML Discussions](https://github.com/bentoml/BentoML/discussions).


## Reporting a Vulnerability

If you discover a potential security vulnerability, we kindly request that you refrain from sharing the information publicly and report it to us directly. Please send an email to security@bentoml.com with the following details:

* Description of the potential vulnerability.
* Steps to reproduce the issue (if applicable).
* Any relevant screenshots or logs.
* Your contact information for further communication.

Alternatively, you can [open a security advisory](https://github.com/bentoml/BentoML/security/advisories/new) on GitHub.

## Default deployment trust model

A BentoML service started with `bentoml serve` listens on `0.0.0.0:3000`
by default. The HTTP and gRPC endpoints accept all requests, and the
inference routes are reachable without authentication.

BentoML does not ship an opinionated authentication layer. The bento
author is expected to add their own protection before exposing the
service to an untrusted network:

- HTTP: mount Starlette middleware that validates an API key, JWT, or
  upstream identity header. Example:

  ```python
  from starlette.middleware import Middleware
  from my_auth import APIKeyMiddleware

  svc.add_asgi_middleware(APIKeyMiddleware, header="x-api-key")
  ```

- gRPC: add an interceptor that enforces the same check at
  `bentoml.grpc.interceptors`.

Reports describing "the default service accepts unauthenticated
requests" therefore match the documented design and will not be
treated as a vulnerability. Reports describing an auth-middleware
bypass once the middleware is in place are in scope.

## Exceptions

The following reports are out of scope and will not be accepted as
security vulnerabilities:

* Reports about pickle-related vulnerabilities in the runner service or
  dependency service. We consider these scenarios to be purely
  theoretical and not a practical vulnerability in BentoML.

BentoML does not participate in Huntr.com's bug bounty program; we have no budget for bug bounties at this time nor do we plan to in the future.
