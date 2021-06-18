FROM centos:${OS_VERSION}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# needed for string substitutions
SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,target=/var/cache/yum \
    yum install -y curl git gcc gcc-c++ make