#!/usr/bin/env bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)

if [[ "$(docker images -q bentoml/yatai-service:dev 2> /dev/null)" != "" ]]; then
  echo "There is no need to rebuild yatai dev docker image. BentoML is installed in editable mode in
  this docker image and local git repo is mounted in the Yatai dev container started with
  'start_dev.sh' script. To try out new changes in Yatai server, simply re-run 'start_dev.sh'
  script."
  echo "To rebuild the yatai server dev image, run 'docker rmi bentoml/yatai-service:dev' and run this script again."
  exit 0
fi

docker build -t bentoml/yatai-service:dev -f- $GIT_ROOT <<EOF
FROM bentoml/yatai-service:latest
WORKDIR /usr/local/bentoml-dev
COPY . .
RUN npm install --global yarn && make install-web-deps && make build-yatai-web-ui
RUN pip install -U -e /usr/local/bentoml-dev
EOF
