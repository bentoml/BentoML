FROM ubuntu:${OS_VERSION}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# needed for string substitutions
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q \
    && apt-get install -q -y --no-install-recommends \
        ca-certificates gnupg2 wget git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*