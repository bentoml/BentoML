FROM amazonlinux:${OS_VERSION} as base-image

# needed for string substitutions
SHELL ["/bin/bash", "-c"]

RUN yum upgrade -y \
    && yum install -y wget ca-certificates \
    && yum clean all -y \
    && rm -rf /var/cache/yum