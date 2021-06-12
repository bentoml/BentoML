#!/bin/sh

# the purpose of this scripts:
# - setup minikube, kubectl, helm if doesn't exists
# - Run through most of configurations setup for k8s cluster.
#
# NOTES:
# - This scripts should run as-is, meaning this script will run within BentoML documents scope.
# - Example BentoService: aarnphm/bentoml-sentiment-analysis:latest
# - Assume that users before running this script to setup VirtualBox
# - This only works on POSIX system

# Contains functions from https://github.com/client9/shlib
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
K8S_CONFIG="$GIT_ROOT/docs/source/guides/configs/deployment"

BINDIR=/usr/local/bin/
LOG_LEVEL=2

is_command() {
	command -v "$1" >/dev/null
}

log_debug() {
	[ 3 -le "${LOG_LEVEL}" ] || return 0
	echo debug "$@" 1>&2
}

log_info() {
	[ 2 -le "${LOG_LEVEL}" ] || return 0
	echo info "$@" 1>&2
}

log_err() {
	[ 1 -le "${LOG_LEVEL}" ] || return 0
	echo error "$@" 1>&2
}

log_crit() {
	[ 0 -le "${LOG_LEVEL}" ] || return 0
	echo critical "$@" 1>&2
}