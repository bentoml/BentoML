#!/usr/bin/env bash

GIT_ROOT="$(git rev-parse --show-toplevel)"

cd "$GIT_ROOT" || exit 1
black .
isort .

# format yatai webui
WEB_SERVER_DIR="$GIT_ROOT"/yatai/web_server
WEB_UI_DIR="$GIT_ROOT"/yatai/ui

cd "$WEB_SERVER_DIR" && yarn format
cd "$WEB_UI_DIR" && yarn format