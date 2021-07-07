#!/usr/bin/env bash

GIT_ROOT="$(git rev-parse --show-toplevel)"

# format yatai webui
WEB_UI_DIR="$GIT_ROOT"/bentoml/yatai/web
cd "$WEB_UI_DIR" && yarn format

# format docker directory format.
DOCKER_DIR="$GIT_ROOT"/docker
isort "$DOCKER_DIR"

black -S "$GIT_ROOT"