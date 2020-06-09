#!/usr/bin/env bash
set -Eeuo pipefail

# check to see if this file is being run or sourced from another script
_is_sourced() {
  # https://unix.stackexchange.com/a/215279
  [ "${#FUNCNAME[@]}" -ge 2 ] \
    && [ "${FUNCNAME[0]}" = '_is_sourced' ] \
    && [ "${FUNCNAME[1]}" = 'source' ]
}

_main() {
  # if first arg looks like a flag, assume we want to start bentoml YataiService
  if [ "${1:0:1}" = '-' ]; then
    set -- bentoml serve-gunicorn "$@" /bento
  fi

  # Set BentoML API server port via env var
  export BENTOML_PORT=$PORT \
  # Backward compatibility for BentoML prior to 0.7.5
  export BENTOML__APISERVER__DEFAULT_PORT=$PORT \

  exec "$@"
}

if ! _is_sourced; then
  _main "$@"
fi