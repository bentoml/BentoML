#!/usr/bin/env bash

fname=$(basename "$0")
dname=$(dirname "$0")

err=0

PYTESTARGS=
GIT_ROOT=$(git rev-parse --show-toplevel)
CONFIG_FILE="$dname/config.yml"
REQ_FILE="/tmp/rq.txt"

cd "$GIT_ROOT" || exit

source "$dname/helpers.sh"

set_on_failed_callback "err=1"


yq_docker() {
  if [[ $(docker images --filter=reference='bentoml/checker' -q) == "" ]]; then
      docker pull bentoml/checker:1.0 || exit
  fi
  docker run -i --rm -v "$GIT_ROOT":/bentoml bentoml/checker:1.0 yq "$@"
}


getval(){
  yq_docker eval "$@" "$CONFIG_FILE"
}


validate_yaml() {
  # validate YAML file
  if ! [ -f "$CONFIG_FILE" ]; then
    FAIL "$CONFIG_FILE does not exists"
    exit 1
  fi

  if ! (yq_docker e --exit-status 'tag == "!!map" or tag== "!!seq"' "$CONFIG_FILE"> /dev/null); then
    FAIL "Invalid YAML file"
    exit 1
  fi
}


usage() {
    cat <<HEREDOC
Running frameworks integration tests with pytest and generate coverage reports. Make sure that given frameworks is defined under $CONFIG_FILE.

Usage:
  $dname/$fname [-h] [-v] <frameworks> <pytest_additional_arguments>

Flags:
  -h            show this message
  -v            set verbost scripts


If pytest_additional_arguments is given, this will be apppended to given tests run.

Example:
  $ $dname/$fname pytorch --gpus
HEREDOC
    exit 2
}


parse_args() {
  while getopts ":vh?:" args; do
    case "${args}" in
      h | \?)
        usage "$0";;
      v)
        set -x;;
      *)
        return 1;;
    esac
  done
  shift $((OPTIND -1))
  PYTESTARGS="${2:-}"
}


parse_config() {
  framework=$1
  test_dir=
  is_dir=
  override_fname=
  extras=

  test_dir=$(getval ".$framework.root_test_dir")
  is_dir=$(getval ".$framework.is_dir")
  override_fname=$(getval ".$framework.override_fname")
  extras=$(getval ".$framework.external_scripts")

  # processing file name
  if [[ "$override_fname" != "" ]]; then
    fname="$override_fname"
  elif [[ "$is_dir" == "false" ]]; then
    fname="test_""$framework""_impl.py"
  else
    fname="$framework"
  fi

  # processing dependencies
  yq_docker eval '.'"$framework"'.dependencies[]' "$CONFIG_FILE" >"$REQ_FILE" || exit
}


main() {
  parse_args "$@"

  validate_yaml
  if [ $OPTIND -eq 1 ]; then
    argv="$@"
  else
    argv="${2:-}"
  fi

  parse_config "$argv"

  OPTS=(--cov=bentoml --cov-config=.coveragerc --cov-report=xml:"$framework.xml")

  if [ -n "$PYTESTARGS" ]; then
    OPTS+=("$PYTESTARGS")
  fi

  # setup tests environment
  pip install -r "$REQ_FILE" && rm "$REQ_FILE"

  if [[ "$extras" != "" ]]; then
    eval "$extras" || exit
  fi

  pytest "$GIT_ROOT"/"$test_dir"/"$fname" "${OPTS[@]}"

  test $err = 0 # Return non-zero if pytest failed

  PASS "$framework integration tests passed!"
}

main "$@"

# vim: set ft=sh ts=2 sw=2 tw=0 et :
