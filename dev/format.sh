#!/usr/bin/env bash

GIT_ROOT="$(git rev-parse --show-toplevel)"

black -S "$GIT_ROOT"
isort "$GIT_ROOT"/bentoml/.

# format docker directory format.
DOCKER_DIR="$GIT_ROOT"/docker
isort "$DOCKER_DIR"

# format yatai webui
WEB_SERVER_DIR="$GIT_ROOT"/yatai/web_server
WEB_UI_DIR="$GIT_ROOT"/yatai/ui

cd "$WEB_SERVER_DIR" && yarn format
cd "$WEB_UI_DIR" && yarn format