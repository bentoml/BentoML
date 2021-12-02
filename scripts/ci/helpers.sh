#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

PASS() {
    echo -e "$GREEN""[PASS]""$NC" "$*"
}

FAIL() {
    echo -e "$RED""[FAIL]""$NC" "$*"
}

INFO() {
    echo -e "$YELLOW""[INFO]""$NC" "$*"
}

set_on_failed_callback() {
    set -E
    trap "$*" ERR
}

check_cmd() {
    command -v "$1" > /dev/null 2>&1
}

need_cmd() {
    if ! check_cmd "$1"; then
        FAIL "need $1 (command not found)"
        exit 1
    fi
}

set -eo pipefail
