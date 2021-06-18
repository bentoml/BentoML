FROM ubuntu:${OS_VERSION} as os-base

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# needed for string substitutions
SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
    ca-certificates gnupg2 curl