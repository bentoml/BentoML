# test platform outputs with buildx:
# docker buildx build -f scripts/docker/platform.Dockerfile --no-cache --progress=plain --platform=<test-platform>
# docker buildx bake -f scripts/docker/docker-bake.hcl platforms

FROM busybox
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETVARIANT
RUN printf "I'm building for TARGETPLATFORM=${TARGETPLATFORM} on BUILDPLATFORM=${BUILDPLATFORM}" \
    && printf ", TARGETARCH=${TARGETARCH}" \
    && printf ", TARGETVARIANT=${TARGETVARIANT} \n" \
    && printf "With uname -s : " && uname -s \
    && printf "and  uname -m : " && uname -m
