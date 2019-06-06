#!/bin/bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
black $GIT_ROOT
