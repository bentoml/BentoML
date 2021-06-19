FROM centos:${OS_VERSION}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# needed for string substitutions
SHELL ["/bin/bash", "-c"]

RUN yum install -y wget git gcc gcc-c++ make \
    && yum clean all \
    && rm -rf /var/cache/yum/*