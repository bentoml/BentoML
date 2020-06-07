#!/usr/bin/env bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
black -S $GIT_ROOT
