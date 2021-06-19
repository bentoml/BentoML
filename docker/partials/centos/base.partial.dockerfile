FROM centos:${OS_VERSION} as base-image

# needed for string substitutions
SHELL ["/bin/bash", "-c"]

RUN yum upgrade -y \
    && yum install -y wget git gcc gcc-c++ ca-certificates make \
    && yum clean all \
    && rm -rf /var/cache/yum/*