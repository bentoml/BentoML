#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)
# format yatai webui
WEB_UI_DIR=$GIT_ROOT/bentoml/yatai/web

black -S "$GIT_ROOT"

cd $WEB_UI_DIR && yarn format

