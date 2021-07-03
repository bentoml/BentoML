# syntax = docker/dockerfile:1.2
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE DO NOT EDIT.
#
# ===========================================


FROM centos:7

# setup ENV and ARG
ENV PYTHONDONTWRITEBYTECODE=1
ARG PYTHON_VERSION

ENV PATH /opt/conda/bin:$PATH

# needed for string substitution
SHELL ["/bin/bash","-exo", "pipefail", "-c"]

RUN yum upgrade -y \
    && yum install -y wget git ca-certificates curl gcc gcc-c++ make \
    && yum clean all \
    && rm -rf /var/cache/yum

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    yum clean all && \
    yum autoremove -y wget && \
    rm -rf /var/cache/yum

RUN /opt/conda/bin/conda install -y python=$PYTHON_VERSION pip && \
    /opt/conda/bin/conda clean -afy

# YataiService GRPC port
EXPOSE 50051

# YataiService Web UI port
EXPOSE 3000

# Install Node, required by the Web UI Server
# https://github.com/nodejs/docker-node/blob/main/16/buster-slim/Dockerfile
ENV NODE_VERSION 16.3.0

# https://github.com/docker-library/faq#openpgp--gnupg-keys-and-verification ?
RUN ARCH='x64' \
    && set -ex \
    && curl -vvv -fsSLO --compressed "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-$ARCH.tar.xz" \
    && tar -xJf "node-v$NODE_VERSION-linux-$ARCH.tar.xz" -C /usr/local --strip-components=1 --no-same-owner \
    && rm "node-v$NODE_VERSION-linux-$ARCH.tar.xz" \
    && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
    # smoke tests
    && node --version \
    && npm --version

RUN yum upgrade -y && yum install -y postgresql-devel && \
    yum clean all && \
    rm -rf /var/cache/yum

COPY tools/bashrc /etc/bash.bashrc
RUN chmod a+r /etc/bash.bashrc