FROM python:slim

RUN pip install mypy-protobuf grpcio-tools~=1.34.0

ENV DEBIAN_FRONTEND noninteractive

# needed for string substitution
SHELL ["/bin/bash","-exo", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -q -y --no-install-recommends \
      ca-certificates curl git build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000
RUN groupadd --gid $GID bentoml \
  && useradd --uid $UID --gid bentoml --shell /bin/bash --create-home bentoml

ENV NVM_DIR /home/nvm
ENV NODE_VERSION 16.3.0
RUN mkdir $NVM_DIR && chown -R "bentoml:bentoml" $NVM_DIR

ENV BENTOML_HOME=/home/bentoml

USER bentoml

RUN mkdir $BENTOML_HOME/workspace

WORKDIR $BENTOML_HOME/workspace

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default \
    # smoke test
    && node --version \
    && npm --version

ENV NODE_PATH $NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV NODE_BIN $NVM_DIR/v$NODE_VERSION/bin
ENV PATH $NODE_BIN:$PATH

RUN . $NVM_DIR/nvm.sh && npm install -g protobufjs@6.9.0