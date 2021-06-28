#!/bin/bash

set -e

# set username and password
UNAME=""
UPASS=""

# get token to be able to talk to Docker Hub
TOKEN=$(curl -s -H "Content-Type: application/json" -X POST -d '{"username": "'${UNAME}'", "password": "'${UPASS}'"}' https://hub.docker.com/v2/users/logins/ | jq -r .token)

curl -s -vvv -X PATCH -H "Content-Type: application/json" -H "Authorization: JWT ${TOKEN}" -d '{"full_description": "Hello world from aarnphms curl archlinux"}' https://hub.docker.com/v2/repositories/aarnphm/model-server/
