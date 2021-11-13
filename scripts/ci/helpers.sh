#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

function PASS() {
    echo -e "$GREEN""[PASS]""$NC" "$*"
}

function FAIL() {
    echo -e "$RED""[FAIL]""$NC" "$*"
}

function set_on_failed_callback() {
    set -E
    # shellcheck disable=SC2064
    trap "$*" ERR
}

set -euo pipefail
