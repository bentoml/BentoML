#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)
black -S "$GIT_ROOT"
