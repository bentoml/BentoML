#!/usr/bin/env bash

set -ex

# set username and password
UNAME="aarnphm"
UPASS=""

alias manager_dockerfiles="docker run --rm -u $(id -u):$(id -g) -v $(pwd):/bentoml bentoml-docker python3 manager.py "

alias manager_images="docker run --rm -v $(pwd):/bentoml -v /var/run/docker.sock:/var/run/docker.sock bentoml-docker python3 manager.py "

# https://github.com/docker/hub-feedback/issues/2006
# get token to be able to talk to Docker Hub
#TOKEN=$(curl -s -H "Content-Type: application/json" -X POST -d '{"username": "'${UNAME}'", "password": "'${UPASS}'"}' https://hub.docker.com/v2/users/logins/ | jq -r .token)
#
#curl -s -vvv -X PATCH -H "Content-Type: application/json" -H "Authorization: JWT ${TOKEN}" -d '{"full_description": "ed"}' https://hub.docker.com/v2/repositories/aarnphm/model-server/