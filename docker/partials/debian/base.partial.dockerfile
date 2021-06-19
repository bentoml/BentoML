FROM debian:${OS_VERSION} as base-image

RUN apt-get update -y \
    && apt-get install -y -q --no-install-recommends build-essential gcc ca-certificates